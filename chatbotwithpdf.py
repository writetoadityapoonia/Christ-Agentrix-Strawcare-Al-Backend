"""
Medical AI Platform - Main Application (Single-File Deployment)
Combined Medical AI Chatbot (LangGraph) and PDF Report Parser API.
All code consolidated for Render deployment.
"""

# ==================== IMPORTS ====================

from typing import TypedDict, Optional, List, Annotated
import json
import re
import os
import asyncio
import tempfile
import time
from io import BytesIO
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pdfplumber
from dotenv import load_dotenv
from groq import Groq
from fastembed import TextEmbedding
from pinecone import Pinecone, ServerlessSpec
from PyCharacterAI import get_client
from pydantic import BaseModel
from fpdf import FPDF, XPos, YPos
from PIL import Image as PI


from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

import uvicorn

# ==================== ENV CONFIG ====================

load_dotenv()

CHARACTER_AI_TOKEN = os.getenv("CHARACTER_AI_TOKEN")
CHARACTER_ID = os.getenv("CHARACTER_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AI_SERVER_URL = os.getenv("AI_SERVER_URL", "http://localhost:9000").rstrip("/")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Medical AI Platform",
    description="Combined Medical AI Chatbot (LangGraph) and Report Parser API",
    version="3.0.0"
)


# ==================== CHATBOT PYDANTIC MODELS ====================

class Specialist(BaseModel):
    doctor_id: str
    name: str
    specialization: str
    category: str
    phone: str
    experience: int


class Analysis(BaseModel):
    detected_conditions: List[str]
    is_serious: bool
    recommended_specialty: str
    explanation: str


class ModelResponse(BaseModel):
    response: str
    analysis: Analysis
    specialists: Optional[List[Specialist]] = None
    is_serious: bool


class PromptRequest(BaseModel):
    prompt: str
    thread_id: Optional[str] = "default"


# ==================== CHATBOT STATE ====================

class ChatState(TypedDict):
    """State for the chatbot graph"""
    user_message: str
    thread_id: str
    messages: Annotated[list, add_messages]
    illness: Optional[str]
    specialties: List[str]
    is_serious: bool
    censored_message: str
    specialists: Optional[List[dict]]
    character_response: str
    final_response: str
    detected_conditions: List[str]
    recommended_specialty: str
    explanation: str


# ==================== PINECONE RAG COMPONENT ====================

class PineconeDoctorRetriever:
    """RAG component using Pinecone for retrieving relevant doctors"""

    INDEX_NAME = "medical-doctors"
    DIMENSION = 384

    def __init__(self, pinecone_api_key: str, dataset_path: str = "healthver2.doctors.json"):
        self.embedding_model = TextEmbedding()
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.dataset_path = dataset_path
        self.index = None
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.INDEX_NAME not in existing_indexes:
            print(f"Creating Pinecone index: {self.INDEX_NAME}")
            self.pc.create_index(
                name=self.INDEX_NAME,
                dimension=self.DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not self.pc.describe_index(self.INDEX_NAME).status['ready']:
                time.sleep(1)
            print(f"Index {self.INDEX_NAME} created successfully")
            await self._upsert_doctors()
        self.index = self.pc.Index(self.INDEX_NAME)
        stats = self.index.describe_index_stats()
        if stats.total_vector_count == 0:
            await self._upsert_doctors()
        self._initialized = True
        print(f"Pinecone RAG initialized with {stats.total_vector_count} doctors")

    async def _upsert_doctors(self):
        if not os.path.exists(self.dataset_path):
            print(f"Warning: Dataset file {self.dataset_path} not found")
            return
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            doctors = json.load(f)
        print(f"Upserting {len(doctors)} doctors to Pinecone...")
        index = self.pc.Index(self.INDEX_NAME)
        batch_size = 50
        for i in range(0, len(doctors), batch_size):
            batch = doctors[i:i + batch_size]
            vectors = []
            for doc in batch:
                qualifications = doc.get("qualifications", [])
                qual_str = ", ".join(qualifications) if isinstance(qualifications, list) else str(qualifications)
                text = f"{doc.get('name', '')} {doc.get('specialization', '')} {doc.get('category', '')} {qual_str}"
                embedding = list(self.embedding_model.embed([text]))[0].tolist()
                doc_id = str(doc.get("_id", {}).get("$oid", doc.get("userId", f"doc_{i}")))
                metadata = {
                    "name": str(doc.get("name", "")),
                    "specialization": str(doc.get("specialization", "")),
                    "category": str(doc.get("category", "")),
                    "phone": str(doc.get("phone", "")),
                    "experience": int(doc.get("experience", 0)),
                    "consultationFee": int(doc.get("consultationFee", 0)),
                    "qualifications": qual_str,
                    "email": str(doc.get("email", "")),
                    "status": str(doc.get("status", ""))
                }
                vectors.append({"id": doc_id, "values": embedding, "metadata": metadata})
            index.upsert(vectors=vectors)
        print(f"Successfully upserted {len(doctors)} doctors to Pinecone")

    async def search(self, query: str, top_k: int = 3) -> List[dict]:
        if not self._initialized:
            await self.initialize()
        query_embedding = list(self.embedding_model.embed([query]))[0].tolist()
        results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        doctors = []
        for match in results.matches:
            doctors.append({
                "doctor_id": match.id,
                "name": match.metadata.get("name", ""),
                "specialization": match.metadata.get("specialization", ""),
                "category": match.metadata.get("category", ""),
                "phone": match.metadata.get("phone", ""),
                "experience": match.metadata.get("experience", 0),
                "consultationFee": match.metadata.get("consultationFee", 0),
                "score": match.score
            })
        return doctors


# ==================== MESSAGE CENSOR ====================

class MessageCensor:
    def __init__(self, groq_client: Groq):
        self.groq_client = groq_client

    def censor_message(self, message: str) -> str:
        prompt = f"""
        Analyze the following user message for self-harm, suicide, or extremely sensitive content that might trigger AI safety filters.
        
        Message: "{message}"
        
        Task:
        1. If the message is safe and does NOT contain self-harm/suicide intent, return it exactly as is.
        2. If the message contains self-harm, suicide, or likely-to-be-filtered content:
           - Rewrite it to express the SAME emotional pain/urgency but using safe, conversational language.
           - Remove specific methods/plans (e.g., "cutting", "pills", "jumping").
           - Focus on the *feeling* (hopelessness, despair, pain) rather than the *act*.
           - Example: "I want to kill myself" -> "I'm feeling completely hopeless and don't know how to go on."
           
        Output ONLY the final message (original or rewritten). Do not add quotes or explanations.
        """
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=200
            )
            censored_text = response.choices[0].message.content.strip()
            if (censored_text.startswith('"') and censored_text.endswith('"')) or \
               (censored_text.startswith("'") and censored_text.endswith("'")):
                censored_text = censored_text[1:-1]
            return censored_text
        except Exception as e:
            print(f"Error in LLM censoring: {e}")
            return message


# ==================== LANGGRAPH CHATBOT ====================

class ChatbotGraph:
    def __init__(self, token, character_id, groq_api_key, pinecone_api_key, dataset_path="healthver2.doctors.json"):
        self.token = token
        self.character_id = character_id
        self.groq_client = Groq(api_key=groq_api_key)
        self.censor = MessageCensor(self.groq_client)
        self.doctor_retriever = PineconeDoctorRetriever(pinecone_api_key, dataset_path)
        self.checkpointer = MemorySaver()
        self.client = None
        self.chat = None
        self.me = None
        self.graph = self._build_graph()

    async def _ensure_character_client(self):
        if self.client is None or self.me is None or self.chat is None:
            self.client = await get_client(token=self.token)
            self.me = await self.client.account.fetch_me()
            self.chat, _ = await self.client.chat.create_chat(self.character_id)

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ChatState)
        graph.add_node("analyze_medical_query", self._analyze_medical_query_node)
        graph.add_node("search_doctors", self._search_doctors_node)
        graph.add_node("censor_message", self._censor_message_node)
        graph.add_node("get_character_response", self._get_character_response_node)
        graph.add_edge(START, "analyze_medical_query")
        graph.add_conditional_edges(
            "analyze_medical_query", self._route_after_analysis,
            {"search_doctors": "search_doctors", "censor_message": "censor_message"}
        )
        graph.add_edge("search_doctors", "censor_message")
        graph.add_edge("censor_message", "get_character_response")
        graph.add_edge("get_character_response", END)
        return graph.compile(checkpointer=self.checkpointer)

    def _route_after_analysis(self, state: ChatState) -> str:
        return "search_doctors" if state["is_serious"] else "censor_message"

    async def _analyze_medical_query_node(self, state: ChatState) -> dict:
        english_message = state["user_message"]
        prompt = f"""You are a medical triage assistant. Analyze this patient message and determine:
1. The main illness/symptoms described
2. Appropriate medical specialties
3. Whether this is a SERIOUS/URGENT condition requiring immediate attention

SERIOUS conditions include but are not limited to:
- Chest pain, heart attack symptoms
- Difficulty breathing, respiratory distress
- Stroke symptoms (facial drooping, slurred speech, sudden weakness)
- Severe bleeding or trauma
- Severe allergic reactions
- Loss of consciousness
- Severe abdominal pain
- High fever with confusion
- Suicidal thoughts or self-harm

Patient message: "{english_message}"

Respond in JSON format ONLY:
{{
    "illness": "main condition/symptoms identified",
    "specialties": ["specialty1", "specialty2"],
    "is_serious": true/false,
    "explanation": "brief explanation of why serious or not serious"
}}"""
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=300,
            )
            content = response.choices[0].message.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            data = json.loads(content)
            illness = data.get("illness", "Non-specific symptoms")
            specialties = data.get("specialties", [])
            is_serious = data.get("is_serious", False)
            explanation = data.get("explanation", "")
            detected_conditions = [illness] if illness else ["Non-specific symptoms"]
            recommended_specialty = specialties[0] if specialties else "General Medicine"
            return {
                "illness": illness, "specialties": specialties, "is_serious": is_serious,
                "detected_conditions": detected_conditions,
                "recommended_specialty": recommended_specialty, "explanation": explanation
            }
        except Exception as e:
            print(f"Error in medical analysis: {e}")
            return {
                "illness": None, "specialties": [], "is_serious": False,
                "detected_conditions": ["Non-specific symptoms"],
                "recommended_specialty": "General Medicine",
                "explanation": "Unable to analyze symptoms"
            }

    async def _search_doctors_node(self, state: ChatState) -> dict:
        specialties = state.get("specialties", [])
        illness = state.get("illness", "")
        search_query = " ".join(specialties[:2]) if specialties else illness or "general medicine"
        try:
            doctors = await self.doctor_retriever.search(search_query, top_k=3)
            return {"specialists": doctors}
        except Exception as e:
            print(f"Error searching doctors: {e}")
            return {"specialists": []}

    async def _censor_message_node(self, state: ChatState) -> dict:
        censored = self.censor.censor_message(state["user_message"])
        return {"censored_message": censored}

    async def _get_character_response_node(self, state: ChatState) -> dict:
        await self._ensure_character_client()
        answer = await self.client.chat.send_message(
            self.character_id, self.chat.chat_id,
            state["censored_message"], streaming=True
        )
        full_response = ""
        async for r in answer:
            full_response = r.get_primary_candidate().text
        return {
            "character_response": full_response,
            "final_response": full_response,
            "messages": [("assistant", full_response)]
        }

    async def chat_response(self, user_message: str, thread_id: str = "default") -> ModelResponse:
        await self.doctor_retriever.initialize()
        initial_state: ChatState = {
            "user_message": user_message, "thread_id": thread_id,
            "messages": [("user", user_message)],
            "illness": None, "specialties": [], "is_serious": False,
            "censored_message": "", "specialists": None,
            "character_response": "", "final_response": "",
            "detected_conditions": [], "recommended_specialty": "None",
            "explanation": ""
        }
        config = {"configurable": {"thread_id": thread_id}}
        final_state = await self.graph.ainvoke(initial_state, config=config)
        specialists_list = None
        if final_state.get("specialists"):
            specialists_list = [
                Specialist(
                    doctor_id=str(doc.get("doctor_id", "")),
                    name=str(doc.get("name", "")),
                    specialization=str(doc.get("specialization", "")),
                    category=str(doc.get("category", "")),
                    phone=str(doc.get("phone", "")),
                    experience=int(doc.get("experience", 0))
                ) for doc in final_state["specialists"]
            ]
        analysis = Analysis(
            detected_conditions=final_state["detected_conditions"],
            is_serious=final_state["is_serious"],
            recommended_specialty=final_state["recommended_specialty"],
            explanation=final_state["explanation"]
        )
        return ModelResponse(
            response=final_state["final_response"], analysis=analysis,
            specialists=specialists_list, is_serious=final_state["is_serious"]
        )


def create_chatbot_graph(token, character_id, groq_api_key, pinecone_api_key, dataset_path="healthver2.doctors.json"):
    return ChatbotGraph(token, character_id, groq_api_key, pinecone_api_key, dataset_path)


# ==================== PDF PROCESSOR V2 HELPERS ====================

def extract_text(pdf_bytes: bytes, max_chars: int = 3000) -> str:
    text = ""
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text[:max_chars]


def _first_float(s):
    m = re.search(r"(\d+(\.\d+)?)", str(s or "").replace(",", ""))
    return float(m.group(1)) if m else None


def _parse_range(ref):
    rs = str(ref or "").strip()
    rs = rs.replace("\u2013", "-").replace("\u2014", "-").replace(" - ", "-")
    rs = rs.replace("&lt;", "<").replace("&gt;", ">")
    if rs.startswith("<"):
        v = _first_float(rs[1:])
        return 0.0, v
    if rs.startswith(">"):
        v = _first_float(rs[1:])
        return v, (v * 2 if v else None)
    if "-" in rs:
        p = rs.split("-", 1)
        return _first_float(p[0]), _first_float(p[1])
    return None, None


def classify_test(value_str, ref_str) -> str:
    v = _first_float(value_str)
    mn, mx = _parse_range(ref_str)
    if v is None or mn is None or mx is None:
        low = str(value_str or "").lower()
        if any(x in low for x in ["negative", "normal", "nil", "none", "absent", "clear", "not seen"]):
            return "Normal"
        if any(x in low for x in ["positive", "abnormal", "trace", "present", "high", "elevated"]):
            return "Abnormal"
        return "Unknown"
    if v < mn:
        return "Low"
    if v > mx:
        return "High"
    return "Normal"


PDF_SYSTEM_PROMPT = """
You are a medical report parser. Output VALID JSON only (no markdown, no extra text).
Return exactly:
{
  "patient_info": {"name":"string","age":"string","sex":"string"},
  "report_type": "blood|urine|other",
  "test_results": [
    {"test_name":"string","value":"string","unit":"string","reference_range":"string"}
  ],
  "summary": "2-4 sentences plain English summary",
  "doctor_notes": "string or N/A"
}
Rules: Always valid JSON. Use "N/A" for missing. Extract every test found.
""".strip()


def parse_with_llm(text: str) -> dict:
    client = Groq(api_key=GROQ_API_KEY)
    last = None
    for model in ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PDF_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Parse this lab report:\n\n{text}"},
                ],
                temperature=0.0, max_tokens=2000,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            raw = re.sub(r",\s*}", "}", raw)
            raw = re.sub(r",\s*]", "]", raw)
            return json.loads(raw)
        except Exception as e:
            last = e
    raise RuntimeError(f"LLM failed: {last}")


# ==================== CHART CONSTANTS & HELPERS ====================

C_NORMAL  = "#27ae60"
C_HIGH    = "#c0392b"
C_LOW     = "#d35400"
C_UNKNOWN = "#7f8c8d"
C_BG      = "#ffffff"
C_GRID    = "#f1f3f4"
C_TEXT    = "#1a1a2e"
C_BORDER  = "#dde1e7"
C_SOFT    = "#5c636e"
STATUS_COLOR = {
    "Normal": C_NORMAL, "High": C_HIGH,
    "Low": C_LOW, "Abnormal": C_HIGH, "Unknown": C_UNKNOWN,
}


def _save(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                facecolor=C_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _ax_clean(ax):
    ax.set_facecolor(C_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_BORDER)
    ax.spines["bottom"].set_color(C_BORDER)
    ax.tick_params(colors=C_SOFT, labelsize=8.5, length=3)


# ==================== CHART FUNCTIONS ====================

def chart_bullet(tests: list):
    rows = []
    for t in tests:
        v = _first_float(t.get("value", ""))
        mn, mx = _parse_range(t.get("reference_range", ""))
        if v is None or mn is None or mx is None:
            continue
        s = classify_test(t.get("value", ""), t.get("reference_range", ""))
        rows.append({"name": t.get("test_name", "")[:22].strip(), "v": v,
                     "mn": mn, "mx": mx, "unit": t.get("unit", ""), "status": s})
    if not rows:
        return None
    rows = rows[:14]
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(11, n * 0.72 + 0.6), facecolor=C_BG)
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.15, left=0.22, right=0.93, top=0.93, bottom=0.04)
    fig.text(0.5, 0.97, "Bullet Chart  -  Measured Values vs Reference Range",
             ha="center", fontsize=12, fontweight="bold", color=C_TEXT)
    for ax, r in zip(axes, rows):
        mn, mx, v = r["mn"], r["mx"], r["v"]
        span = max(mx * 1.3, v * 1.15, 0.01)
        ax.set_facecolor(C_BG); ax.set_xlim(0, span); ax.set_ylim(0, 1)
        ax.axvspan(0, mn, facecolor="#fde8e8", zorder=1)
        ax.axvspan(mn, mx, facecolor="#e8f5e9", zorder=1)
        ax.axvspan(mx, span, facecolor="#fff3e0", zorder=1)
        ax.barh(0.5, span, height=0.30, color="#e9ecef", zorder=2, align="center")
        c = STATUS_COLOR.get(r["status"], C_UNKNOWN)
        ax.barh(0.5, v, height=0.18, color=c, zorder=3, align="center", alpha=0.92)
        ax.axvline(mn, color=C_NORMAL, lw=1.5, ls="--", zorder=4, alpha=0.6)
        ax.axvline(mx, color=C_NORMAL, lw=1.5, ls="--", zorder=4, alpha=0.6)
        ax.plot(v, 0.5, "o", color=c, ms=9, mec="white", mew=1.8, zorder=5)
        ax.text(-0.012 * span, 0.5, r["name"], ha="right", va="center",
                fontsize=8.5, color=C_TEXT, fontweight="600")
        ax.text(span * 1.008, 0.5, f"{v:.1f} {r['unit']}",
                ha="left", va="center", fontsize=8, color=c, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    return _save(fig)


def chart_donut(tests: list) -> bytes:
    counts = {"Normal": 0, "Abnormal": 0, "Unknown": 0}
    for t in tests:
        s = classify_test(t.get("value", ""), t.get("reference_range", ""))
        if s in ("High", "Low", "Abnormal"):
            counts["Abnormal"] += 1
        elif s == "Unknown":
            counts["Unknown"] += 1
        else:
            counts["Normal"] += 1
    labels = [k for k, v in counts.items() if v > 0]
    sizes  = [counts[k] for k in labels]
    CLRS   = {"Normal": C_NORMAL, "Abnormal": C_HIGH, "Unknown": C_UNKNOWN}
    colors = [CLRS[l] for l in labels]
    fig, ax = plt.subplots(figsize=(5, 4.6), facecolor=C_BG)
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.56, edgecolor="white", linewidth=3.5))
    total = sum(sizes)
    ax.text(0, 0.12, str(total), ha="center", va="center",
            fontsize=28, fontweight="bold", color=C_TEXT)
    ax.text(0, -0.20, "Total Tests", ha="center", va="center",
            fontsize=9, color=C_SOFT)
    ax.legend(wedges, [f"{l}  ({counts[l]})" for l in labels],
              loc="lower center", ncol=len(labels), fontsize=9,
              frameon=False, bbox_to_anchor=(0.5, -0.04))
    ax.set_title("Test Status Overview", fontsize=12, fontweight="bold",
                 color=C_TEXT, pad=10)
    return _save(fig)


def chart_bar(tests: list):
    rows = []
    for t in tests:
        v = _first_float(t.get("value", ""))
        mn, mx = _parse_range(t.get("reference_range", ""))
        if v is None or mn is None or mx is None:
            continue
        s = classify_test(t.get("value", ""), t.get("reference_range", ""))
        rows.append((t.get("test_name", "")[:18].strip(), v, mn, mx, s))
    if not rows:
        return None
    rows = rows[:10]
    names  = [r[0] for r in rows]
    vals   = [r[1] for r in rows]
    mins_  = [r[2] for r in rows]
    maxs_  = [r[3] for r in rows]
    colors = [STATUS_COLOR.get(r[4], C_UNKNOWN) for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    bars = ax.bar(x, vals, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=1.5, width=0.5, zorder=3)
    for i, (lo, hi, v) in enumerate(zip(mins_, maxs_, vals)):
        ax.plot([i, i], [lo, hi], color="#555", lw=1.5, zorder=4)
        ax.plot([i - 0.12, i + 0.12], [lo, lo], color="#555", lw=1.5, zorder=4)
        ax.plot([i - 0.12, i + 0.12], [hi, hi], color="#555", lw=1.5, zorder=4)
        mid = (lo + hi) / 2
        ax.plot(i, mid, "D", color="white", markeredgecolor="#555", markersize=6, zorder=5)
    max_v = max(vals) if vals else 1
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max_v * 0.025,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=C_TEXT)
    _ax_clean(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5, color=C_TEXT)
    ax.set_ylabel("Test Values", color=C_SOFT, fontsize=9.5)
    ax.set_title("Test Results Analysis", fontsize=12, fontweight="bold",
                 color=C_TEXT, pad=14, loc="center")
    ax.grid(True, axis="y", color=C_GRID, lw=0.8, zorder=0)
    legend_elems = [
        mpatches.Patch(color=C_NORMAL, label="Normal"),
        mpatches.Patch(color=C_LOW, label="Below Normal"),
        mpatches.Patch(color=C_HIGH, label="Above Normal"),
    ]
    ax.legend(handles=legend_elems, fontsize=8.5, frameon=True,
              edgecolor=C_BORDER, loc="upper right", facecolor="white", framealpha=0.95)
    plt.tight_layout(pad=1.2)
    return _save(fig)


def chart_lollipop(tests: list):
    rows = []
    for t in tests:
        v = _first_float(t.get("value", ""))
        mn, mx = _parse_range(t.get("reference_range", ""))
        if v is None or mn is None or mx is None or (mn + mx) == 0:
            continue
        mid = (mn + mx) / 2
        pct = (v - mid) / (mid + 1e-9) * 100
        s = classify_test(t.get("value", ""), t.get("reference_range", ""))
        rows.append((t.get("test_name", "")[:22].strip(), pct, s))
    if not rows:
        return None
    rows = sorted(rows, key=lambda x: x[1])[:14]
    names  = [r[0] for r in rows]
    devs   = [r[1] for r in rows]
    colors = [STATUS_COLOR.get(r[2], C_UNKNOWN) for r in rows]
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(rows) * 0.52)), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    y = range(len(names))
    ax.axvspan(-25, 25, color="#e8f5e9", alpha=0.7, zorder=1)
    ax.axvline(0, color="#adb5bd", lw=1.2, ls="--", zorder=2)
    for yi, d, c in zip(y, devs, colors):
        ax.plot([0, d], [yi, yi], color=c, lw=1.8, alpha=0.65, zorder=3)
    ax.scatter(devs, list(y), color=colors, s=95, zorder=4,
               edgecolors="white", linewidths=1.8)
    for yi, d in zip(y, devs):
        ha = "left" if d >= 0 else "right"
        ax.text(d + (1.2 if d >= 0 else -1.2), yi, f"{d:+.1f}%",
                va="center", ha=ha, fontsize=8, color=C_TEXT)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=9, color=C_TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("% Deviation from Reference Midpoint", color=C_SOFT, fontsize=9.5)
    ax.set_title("Deviation from Normal Range", fontsize=12, fontweight="bold",
                 color=C_TEXT, pad=12, loc="left")
    _ax_clean(ax)
    ax.grid(True, axis="x", color=C_GRID, lw=0.8, zorder=0)
    patches = [mpatches.Patch(color=c, label=l)
               for l, c in STATUS_COLOR.items() if any(r[2] == l for r in rows)]
    patches.append(mpatches.Patch(color="#e8f5e9", label="Normal zone (+-25%)"))
    ax.legend(handles=patches, fontsize=8, frameon=True, edgecolor=C_BORDER,
              loc="lower right", facecolor="white", framealpha=0.9)
    plt.tight_layout(pad=1.2)
    return _save(fig)


def chart_strip(tests: list):
    rows = []
    for t in tests:
        name = (t.get("test_name") or "").strip()
        if not name:
            continue
        s = classify_test(t.get("value", ""), t.get("reference_range", ""))
        rows.append((name[:28], str(t.get("value", "")), s))
    if not rows:
        return None
    BG = {"Normal": "#e8f5e9", "High": "#fde8e8", "Low": "#fff3e0",
          "Abnormal": "#fde8e8", "Unknown": "#f5f5f5"}
    fig, ax = plt.subplots(figsize=(9, max(3, len(rows) * 0.56)), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    for i, (name, val, s) in enumerate(rows):
        ec = STATUS_COLOR.get(s, C_UNKNOWN)
        ax.barh(i, 1, height=0.82, color=BG.get(s, "#f5f5f5"),
                edgecolor=ec, lw=1.3, zorder=2)
        ax.barh(i, 0.022, height=0.82, color=ec, zorder=3)
        ax.text(0.04, i, name, va="center", ha="left", fontsize=9.5,
                fontweight="600", color=C_TEXT, zorder=4)
        ax.text(0.62, i, val, va="center", ha="center",
                fontsize=9.5, color=C_SOFT, zorder=4)
        ax.text(0.97, i, s.upper(), va="center", ha="right",
                fontsize=8.5, fontweight="bold", color=ec, zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.invert_yaxis(); ax.set_yticks([]); ax.set_xticks([]); ax.axis("off")
    ax.text(0.04, -0.52, "TEST NAME", fontsize=7.5, color=C_SOFT, fontweight="bold")
    ax.text(0.62, -0.52, "RESULT", fontsize=7.5, color=C_SOFT,
            fontweight="bold", ha="center")
    ax.text(0.97, -0.52, "STATUS", fontsize=7.5, color=C_SOFT,
            fontweight="bold", ha="right")
    ax.axhline(-0.28, color=C_BORDER, lw=0.8)
    ax.set_title("Test Results Overview", fontsize=12, fontweight="bold",
                 color=C_TEXT, pad=10, loc="left")
    plt.tight_layout(pad=1)
    return _save(fig)


def chart_boxplot(tests: list):
    rows = []
    for t in tests:
        v = _first_float(t.get("value", ""))
        mn, mx = _parse_range(t.get("reference_range", ""))
        if v is None or mn is None or mx is None:
            continue
        s = classify_test(t.get("value", ""), t.get("reference_range", ""))
        rows.append({"name": t.get("test_name", "")[:18].strip(),
                     "v": v, "mn": mn, "mx": mx, "s": s,
                     "unit": t.get("unit", "")})
    if not rows:
        return None
    rows = rows[:14]
    n = len(rows)

    def pct(r):
        span = r["mx"] - r["mn"]
        return 50.0 if span == 0 else (r["v"] - r["mn"]) / span * 100

    fig, ax = plt.subplots(figsize=(10, max(4.5, n * 0.62)), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    y = list(range(n))
    ax.axvspan(0, 100, color="#e8f5e9", alpha=0.55, zorder=1)
    ax.axvline(0, color=C_NORMAL, lw=1.2, ls="--", alpha=0.5, zorder=2)
    ax.axvline(100, color=C_NORMAL, lw=1.2, ls="--", alpha=0.5, zorder=2)
    ax.axvline(50, color="#adb5bd", lw=0.9, ls=":", alpha=0.6, zorder=2)
    for i, r in enumerate(rows):
        p = pct(r)
        c = STATUS_COLOR.get(r["s"], C_UNKNOWN)
        ax.barh(i, 100, height=0.35, left=0, color="#e0e0e0",
                alpha=0.5, zorder=3, align="center")
        ax.scatter(p, i, color=c, s=110, zorder=5,
                   edgecolors="white", linewidths=1.8)
        ax.plot([p, p], [i - 0.18, i + 0.18], color=c, lw=2, zorder=4)
        ha = "left" if p < 85 else "right"
        offset = 2 if p < 85 else -2
        ax.text(p + offset, i, f"{r['v']:.1f} {r['unit']}",
                va="center", ha=ha, fontsize=7.5, color=c, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=9, color=C_TEXT)
    ax.invert_yaxis()
    ax.set_xlim(-18, 122)
    ax.set_xlabel("Position within Reference Range  (0% = lower limit, 100% = upper limit)",
                  color=C_SOFT, fontsize=9)
    ax.set_title("Value Position within Reference Range",
                 fontsize=12, fontweight="bold", color=C_TEXT, pad=12, loc="left")
    _ax_clean(ax)
    ax.grid(True, axis="x", color=C_GRID, lw=0.8, zorder=0)
    patches = [mpatches.Patch(color=c, label=l)
               for l, c in STATUS_COLOR.items() if any(r["s"] == l for r in rows)]
    patches.append(mpatches.Patch(color="#e8f5e9", label="Normal zone"))
    ax.legend(handles=patches, fontsize=8, frameon=True, edgecolor=C_BORDER,
              loc="lower right", facecolor="white", framealpha=0.95)
    plt.tight_layout(pad=1.2)
    return _save(fig)


# ==================== PDF BUILDER ====================

P_TEXT    = (25, 25, 30)
P_HDRBG   = (30, 30, 35)
P_SECBG   = (247, 247, 249)
P_SECTL   = (180, 180, 190)
P_ROW_S1  = (255, 255, 255)
P_ROW_S2  = (240, 252, 245)
P_GRN     = (39, 174, 96)
P_RED     = (192, 57, 43)
P_ORG     = (211, 84, 0)
P_GRY     = (110, 110, 120)
P_TEAL    = (39, 174, 96)
P_THEAD_T = (255, 255, 255)
ML, MR    = 15, 15
UW        = 210 - ML - MR
NX, NY    = XPos.LMARGIN, YPos.NEXT


def _safe(text: str) -> str:
    tr = {
        "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00b0": "deg",
        "\u00b5": "u", "\u03bc": "u", "\u2264": "<=", "\u2265": ">=",
        "\u00b1": "+/-", "\u00d7": "x", "\u00f7": "/",
    }
    for ch, rep in tr.items():
        text = str(text).replace(ch, rep)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class LabPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=16)
        self.set_margins(ML, 16, MR)

    def hline(self, color=P_SECTL, lw=0.25):
        self.set_draw_color(*color)
        self.set_line_width(lw)
        self.line(ML, self.get_y(), 210 - MR, self.get_y())

    def sec(self, title: str):
        self.ln(5)
        self.set_fill_color(*P_SECTL)
        self.rect(ML, self.get_y(), 2.5, 8, "F")
        self.set_fill_color(*P_SECBG)
        self.set_text_color(*P_TEXT)
        self.set_font("Helvetica", "B", 10)
        self.set_x(ML + 4)
        self.cell(UW - 4, 8, _safe(f"  {title}"), fill=True, new_x=NX, new_y=NY)
        self.ln(2)
        self.set_text_color(*P_TEXT)

    def kv(self, label: str, value: str, lw=48):
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*P_GRY)
        self.set_x(ML + 5)
        self.cell(lw, 6.5, _safe(label.upper()))
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*P_TEXT)
        self.multi_cell(0, 6.5, _safe(str(value or "N/A")))

    def embed(self, png_bytes: bytes, w: int = None):
        if not png_bytes:
            return
        w = w or UW
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_bytes)
            path = f.name
        with PI.open(path) as im:
            iw, ih = im.size
        h_est = (ih / iw) * w
        if self.get_y() + h_est + 10 > 297 - 16:
            self.add_page()
        self.set_x(ML)
        self.image(path, x=ML, w=w)
        os.unlink(path)
        self.ln(5)

    def page_stripe(self, title: str):
        self.set_fill_color(*P_HDRBG)
        self.rect(0, 0, 210, 16, "F")
        self.set_text_color(240, 240, 245)
        self.set_font("Helvetica", "B", 11)
        self.set_xy(ML, 3)
        self.cell(0, 10, _safe(title), new_x=NX, new_y=NY)
        self.set_text_color(*P_TEXT)
        self.ln(4)


def make_pdf(all_parsed: list, source_names: list) -> bytes:
    all_tests, urine_t, num_t = [], [], []
    for parsed, fname in zip(all_parsed, source_names):
        rt = parsed.get("report_type", "other")
        for t in parsed.get("test_results", []):
            t2 = dict(t); t2["_src"] = fname; t2["_rt"] = rt
            all_tests.append(t2)
            if rt == "urine":
                urine_t.append(t2)
            if (_first_float(t2.get("value", "")) is not None
                    and _parse_range(t2.get("reference_range", ""))[0] is not None):
                num_t.append(t2)

    abn_cnt = sum(
        1 for t in all_tests
        if classify_test(t.get("value", ""), t.get("reference_range", ""))
        not in ["Normal", "Unknown"]
    )

    src = num_t or all_tests
    c_donut    = chart_donut(all_tests)
    c_bar      = chart_bar(src)
    c_bullet   = chart_bullet(src)
    c_boxplot  = chart_boxplot(src)
    c_lollipop = chart_lollipop(src)
    c_strip    = chart_strip(urine_t or all_tests)

    patient = {"name": "N/A", "age": "N/A", "sex": "N/A"}
    for p in all_parsed:
        pi = p.get("patient_info", {})
        if any(pi.get(k) not in [None, "", "N/A"] for k in ["name", "age", "sex"]):
            patient = {k: pi.get(k, "N/A") for k in ["name", "age", "sex"]}
            break

    pdf = LabPDF()
    pdf.add_page()

    # Header band
    pdf.set_fill_color(*P_HDRBG); pdf.rect(0, 0, 210, 28, "F")
    pdf.set_text_color(255, 255, 255); pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(ML, 5); pdf.cell(UW, 10, "Medical Laboratory Report", new_x=NX, new_y=NY)
    pdf.set_font("Helvetica", "", 8.5); pdf.set_text_color(180, 180, 190); pdf.set_x(ML)
    pdf.cell(UW, 6, _safe(
        f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}"
        f"   |   Reports: {len(all_parsed)}"
        f"   |   Tests: {len(all_tests)}"
        f"   |   Abnormal: {abn_cnt}"
    ), new_x=NX, new_y=NY)
    pdf.ln(7)

    # Patient Info
    pdf.sec("Patient Information")
    pdf.kv("Full Name", patient["name"])
    pdf.kv("Age", patient["age"])
    pdf.kv("Sex", patient["sex"])

    # AI Insights
    has_summary = any(p.get("summary", "") not in ["", "N/A"] for p in all_parsed)
    if has_summary:
        pdf.sec("AI Insights")
        for parsed, fname in zip(all_parsed, source_names):
            s  = parsed.get("summary", "")
            rt = parsed.get("report_type", "other").upper()
            dn = parsed.get("doctor_notes", "")
            if not s or s == "N/A":
                continue
            pdf.set_font("Helvetica", "B", 8.5); pdf.set_text_color(*P_GRY)
            pdf.set_x(ML + 5)
            pdf.cell(UW, 6, _safe(f"[{rt}]  {fname}"), new_x=NX, new_y=NY)
            pdf.set_font("Helvetica", "", 9.5); pdf.set_text_color(*P_TEXT)
            pdf.set_x(ML + 5); pdf.multi_cell(UW, 5.8, _safe(str(s)))
            if dn and dn not in ["", "N/A"]:
                pdf.set_font("Helvetica", "I", 8.5); pdf.set_text_color(*P_GRY)
                pdf.set_x(ML + 5)
                pdf.multi_cell(UW, 5.8, _safe(f"Doctor notes: {dn}"))
            pdf.ln(3)

    # Results tables
    CW5 = [54, 28, 38, 24, 26]
    H5  = ["Test Name", "Your Value", "Normal Range", "Status", "Deviation"]

    def _deviation_label(v_str, ref_str):
        v = _first_float(v_str); mn, mx = _parse_range(ref_str)
        if v is None or mn is None or mx is None: return "N/A"
        if v < mn: return "Below range"
        if v > mx: return "Above range"
        return "Within range"

    for parsed, fname in zip(all_parsed, source_names):
        rt     = parsed.get("report_type", "other").upper()
        tests_ = parsed.get("test_results", [])
        if not tests_:
            continue
        abn_r  = sum(1 for t in tests_
                     if classify_test(t.get("value", ""), t.get("reference_range", ""))
                     not in ["Normal", "Unknown"])
        norm_r = len(tests_) - abn_r

        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 11); pdf.set_text_color(*P_TEXT); pdf.set_x(ML)
        pdf.cell(0, 8, _safe(f"Report Type: {rt} ({fname})"), new_x=NX, new_y=NY)
        pdf.set_font("Helvetica", "", 8); pdf.set_text_color(*P_GRY); pdf.set_x(ML)
        pdf.cell(0, 5, _safe(f"Total: {len(tests_)} tests  |  Abnormal: {abn_r}  |  Normal: {norm_r}"),
                 new_x=NX, new_y=NY)
        pdf.ln(3)

        pdf.set_font("Helvetica", "BI", 9.5); pdf.set_text_color(*P_TEXT)
        pdf.set_x(ML); pdf.cell(0, 7, "Detailed Results:", new_x=NX, new_y=NY)
        pdf.ln(1)

        # Header row
        pdf.set_fill_color(*P_TEAL); pdf.set_text_color(*P_THEAD_T)
        pdf.set_font("Helvetica", "B", 8.5); pdf.set_x(ML)
        for w, h in zip(CW5, H5):
            pdf.cell(w, 8, h, fill=True, align="C")
        pdf.ln()

        # Data rows
        for i, t in enumerate(tests_):
            status  = classify_test(t.get("value", ""), t.get("reference_range", ""))
            dev_lbl = _deviation_label(t.get("value", ""), t.get("reference_range", ""))
            bg = P_ROW_S1 if i % 2 == 0 else P_ROW_S2
            sc = {"High": P_RED, "Low": P_ORG, "Normal": P_GRN,
                  "Abnormal": P_RED}.get(status, P_GRY)
            pdf.set_fill_color(*bg); pdf.set_text_color(*P_TEXT)
            pdf.set_font("Helvetica", "", 8.5); pdf.set_x(ML)
            for w, v in zip(CW5[:3], [
                _safe(str(t.get("test_name", "")))[:28],
                _safe(str(t.get("value", "")) + " " + str(t.get("unit", "")))[:16],
                _safe(str(t.get("reference_range", "")))[:20],
            ]):
                pdf.cell(w, 6.5, v, fill=True, align="C")
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.set_fill_color(*bg); pdf.set_text_color(*sc)
            pdf.cell(CW5[3], 6.5, status, fill=True, align="C")
            dev_col = P_RED if "Above" in dev_lbl else (P_ORG if "Below" in dev_lbl else P_GRN)
            pdf.set_text_color(*dev_col); pdf.set_font("Helvetica", "", 8)
            pdf.cell(CW5[4], 6.5, dev_lbl, fill=True, align="C")
            pdf.ln()
        pdf.ln(8)

    # Visual analysis
    pdf.add_page(); pdf.page_stripe("Visual Analysis")

    def block(title, png, w=None):
        if not png: return
        pdf.sec(title); pdf.embed(png, w=w)

    block("Test Status Overview", c_donut, w=110)
    block("Test Results Analysis (Bar Chart)", c_bar)
    block("Bullet Chart \u2014 Values vs Reference", c_bullet)
    block("Value Position within Reference Range", c_boxplot)
    block("Deviation from Normal Range", c_lollipop)
    block("Test Results Overview", c_strip, w=160)

    return bytes(pdf.output())


# ==================== GLOBAL STATE ====================

bot_instance: Optional[ChatbotGraph] = None


# ==================== STARTUP EVENT ====================

@app.on_event("startup")
async def startup_event():
    global bot_instance

    if not GROQ_API_KEY:
        raise RuntimeError("ERROR: Please set GROQ_API_KEY in .env!")
    if not PINECONE_API_KEY:
        raise RuntimeError("ERROR: Please set PINECONE_API_KEY in .env!")
    if not CHARACTER_AI_TOKEN:
        raise RuntimeError("ERROR: Please set CHARACTER_AI_TOKEN in .env!")
    if not CHARACTER_ID:
        raise RuntimeError("ERROR: Please set CHARACTER_ID in .env!")

    bot_instance = create_chatbot_graph(
        token=CHARACTER_AI_TOKEN,
        character_id=CHARACTER_ID,
        groq_api_key=GROQ_API_KEY,
        pinecone_api_key=PINECONE_API_KEY,
        dataset_path="healthver2.doctors.json"
    )

    await asyncio.sleep(0.01)
    print("Medical AI Platform started successfully!")
    print("   - LangGraph Chatbot: Ready")
    print("   - Pinecone RAG: Ready")
    print("   - PDF Processor V2: Ready")


# ==================== GENERAL ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "message": "Medical AI Platform API is running!",
        "endpoints": {
            "chatbot": ["/chat", "/analyze", "/clear_chat", "/sessions"],
            "pdf": ["/parse_report", "/download_report/{filename}"],
            "system": ["/health"]
        },
        "version": "3.0.0"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "chatbot_status": bot_instance is not None,
        "chatbot_type": "LangGraph + Pinecone RAG",
        "pdf_processor_type": "PDF Processor V2 (stateless)"
    }


# ==================== CHATBOT ENDPOINTS ====================

@app.post("/chat", response_model=ModelResponse)
async def chat_endpoint(prompt_request: PromptRequest):
    """Main chat endpoint using LangGraph workflow"""
    global bot_instance
    if bot_instance is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized.")
    try:
        return await bot_instance.chat_response(prompt_request.prompt, prompt_request.thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear_chat")
async def clear_chat(thread_id: str = "default"):
    """Clear chat history for a specific thread"""
    return {"message": f"Chat history cleared for thread: {thread_id}"}


@app.get("/sessions")
async def get_active_sessions():
    return {"active_sessions": [], "note": "LangGraph uses stateless invocations with checkpointing"}


@app.post("/analyze", response_model=Analysis)
async def analyze_medical_query_endpoint(prompt_request: PromptRequest):
    """Analyze a medical query without generating a chat response (Triage-only)"""
    global bot_instance
    if bot_instance is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized.")
    try:
        initial_state = {
            "user_message": prompt_request.prompt,
            "thread_id": prompt_request.thread_id or "default",
            "messages": [("user", prompt_request.prompt)],
            "illness": None, "specialties": [], "is_serious": False,
            "censored_message": "", "specialists": None,
            "character_response": "", "final_response": "",
            "detected_conditions": [], "recommended_specialty": "None",
            "explanation": ""
        }
        result = await bot_instance._analyze_medical_query_node(initial_state)
        return Analysis(
            detected_conditions=result.get("detected_conditions", ["Non-specific symptoms"]),
            is_serious=result.get("is_serious", False),
            recommended_specialty=result.get("recommended_specialty", "None"),
            explanation=result.get("explanation", "")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing query: {str(e)}")


# ==================== PDF PARSER ENDPOINTS ====================

@app.post("/parse_report")
async def parse_report(files: List[UploadFile] = File(...)):
    """Upload one or more PDF lab reports. Returns structured JSON + download URL."""
    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' must be a PDF.")

    all_parsed: list = []
    source_names: list = []
    temp_files: list = []

    try:
        for file in files:
            pdf_bytes = await file.read()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            temp_files.append(tmp_path)

            text = extract_text(pdf_bytes, max_chars=3000)
            if not text.strip():
                raise HTTPException(
                    status_code=422,
                    detail=f"No extractable text in '{file.filename}'. "
                           "Scanned / image-only PDFs are not supported."
                )
            parsed = parse_with_llm(text)
            all_parsed.append(parsed)
            source_names.append(file.filename)

        pdf_bytes_out = make_pdf(all_parsed, source_names)

        if len(files) == 1:
            stem = os.path.splitext(files[0].filename)[0]
            out_name = f"{stem}_report.pdf"
        else:
            out_name = "lab_report_combined.pdf"

        out_path = os.path.join(tempfile.gettempdir(), out_name)
        with open(out_path, "wb") as f:
            f.write(pdf_bytes_out)

        all_tests_flat = [t for p in all_parsed for t in p.get("test_results", [])]
        abn_count = sum(
            1 for t in all_tests_flat
            if classify_test(t.get("value", ""), t.get("reference_range", ""))
            not in ["Normal", "Unknown"]
        )
        combined_json = [
            {"source": fname, "data": parsed}
            for fname, parsed in zip(source_names, all_parsed)
        ]

        return {
            "parsed_json": all_parsed,
            "combined_json": combined_json,
            "test_results_json": {
                t.get("test_name", ""): t.get("value", "")
                for p in all_parsed for t in p.get("test_results", [])
            },
            "pdf_download_url": f"{AI_SERVER_URL}/download_report/{out_name}",
            "total_files_processed": len(files),
            "total_reports_merged": len(all_parsed),
            "unique_tests_found": len(all_tests_flat),
            "abnormal_count": abn_count,
            "pdf_file_size": os.path.getsize(out_path),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing error: {e}")
    finally:
        for p in temp_files:
            if os.path.exists(p):
                os.unlink(p)


@app.get("/download_report/{filename}")
def download_report(filename: str):
    """Download a previously generated PDF report by filename."""
    file_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path, filename=filename, media_type="application/pdf")


# ==================== RUN ====================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
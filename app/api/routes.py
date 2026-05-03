"""
API route'ları — Modül 2 güncellemesi: LLM intent detection eklendi.
"""
from fastapi import APIRouter, HTTPException  # FastAPI bileşenleri
from pydantic import BaseModel  # Veri şeması ve validasyon için
import httpx  # Asenkron HTTP istemcisi

from app.core.config import settings
from app.llm.intent_detector import intent_detector
from app.schemas.intent import IntentResult


router = APIRouter()  # API yönlendirici oluşturur


class HealthResponse(BaseModel):  # Sağlık durumu yanıt şeması
    api: str
    database: str
    llm: str
    vector_store: str


class QueryRequest(BaseModel):  # Kullanıcı sorgu isteği şeması
    query: str
    max_results: int = 5


@router.get("/health", response_model=HealthResponse, tags=["health"])  # GET route endpoint
async def health_check() -> HealthResponse:
    """Sistem bileşenlerinin sağlık durumunu döner."""
    llm_status = "offline"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            if r.status_code == 200:
                llm_status = "online"
    except Exception:
        pass

    return HealthResponse(
        api="online",
        database="not_connected",
        llm=llm_status,
        vector_store="not_initialized",
    )


@router.post("/query", response_model=IntentResult, tags=["assistant"])  # POST route endpoint
async def process_query(request: QueryRequest) -> IntentResult:
    """
    Kullanıcı sorgusunu LLM ile analiz eder.
    Intent ve filtreleri yapılandırılmış JSON olarak döner.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Sorgu boş olamaz.")

    try:
        result = await intent_detector.detect(request.query)
        return result
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="LLM servisi (Ollama) çalışmıyor. 'ollama serve' komutunu çalıştırın.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
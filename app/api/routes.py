"""
API route'ları — Modül 3 güncellemesi: DB sağlık kontrolü eklendi.
"""
from fastapi import APIRouter, Depends, HTTPException  # FastAPI bileşenleri
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text  # SQLAlchemy ORM bileşenleri
import httpx  # Asenkron HTTP istemcisi

from app.core.config import settings
from app.db.session import get_db
from app.llm.intent_detector import intent_detector
from app.schemas.intent import IntentResult

router = APIRouter()  # API yönlendirici oluşturur


@router.get("/health", tags=["health"])  # GET route endpoint
async def health_check(db: AsyncSession = Depends(get_db)) -> dict:
    """Sistem bileşenlerinin sağlık durumunu döner."""
    # LLM kontrolü
    llm_status = "offline"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            if r.status_code == 200:
                llm_status = "online"
    except Exception:
        pass

    # Veritabanı kontrolü
    db_status = "disconnected"
    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        pass

    return {
        "api": "online",
        "database": db_status,
        "llm": llm_status,
        "vector_store": "not_initialized",
    }


@router.post("/query", response_model=IntentResult, tags=["assistant"])  # POST route endpoint
async def process_query(
    request: dict,
    db: AsyncSession = Depends(get_db),
) -> IntentResult:
    """Kullanıcı sorgusunu analiz eder."""
    query = request.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Sorgu boş olamaz.")
    try:
        return await intent_detector.detect(query)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="LLM servisi çalışmıyor.")


@router.get("/products", tags=["products"])  # GET route endpoint
async def list_products(
    db: AsyncSession = Depends(get_db),
    limit: int = 10,
    offset: int = 0,
) -> dict:
    """Ürünleri sayfalı olarak listeler."""
    from sqlalchemy import select  # SQLAlchemy ORM bileşenleri
    from app.db.models import Product

    result = await db.execute(
        select(Product).where(Product.is_active == True).limit(limit).offset(offset)
    )
    products = result.scalars().all()
    return {
        "items": [
            {"id": str(p.id), "name": p.name, "brand": p.brand,
             "price": p.price, "category": p.category}
            for p in products
        ],
        "count": len(products),
    }
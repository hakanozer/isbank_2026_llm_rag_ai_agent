"""
API route'ları — Modül 4: RAG pipeline entegre edildi.
"""
from fastapi import APIRouter, Depends, HTTPException, logger  # FastAPI bileşenleri
from pydantic import BaseModel  # Veri şeması ve validasyon için
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text  # SQLAlchemy ORM bileşenleri
import httpx  # Asenkron HTTP istemcisi

from app.core.config import settings
from app.db.session import get_db
from app.rag.pipeline import rag_pipeline
from app.rag.vector_store import vector_store

router = APIRouter()  # API yönlendirici oluşturur


class QueryRequest(BaseModel):  # Kullanıcı sorgu isteği şeması
    query: str
    max_results: int = 5


class ProductResult(BaseModel):
    product_id: str
    name: str
    brand: str
    price: float
    similarity_score: float


class QueryResponse(BaseModel):  # Sorgu yanıt şeması
    answer: str
    intent: str
    products: list[ProductResult]
    total_found: int


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


@router.post("/query", response_model=QueryResponse, tags=["assistant"])  # POST route endpoint
async def process_query(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """
    Kullanıcı sorgusunu RAG pipeline ile işler.
    Intent detection → Vector search → DB retrieval → LLM yanıt
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Sorgu boş olamaz.")

    try:
        result = await rag_pipeline.run(
            query=request.query,
            db=db,
            top_k=request.max_results,
        )
        return QueryResponse(
            answer=result.answer,
            intent=result.intent.intent.value,
            products=[
                ProductResult(
                    product_id=p.product_id,
                    name=p.name,
                    brand=p.brand,
                    price=p.price,
                    similarity_score=round(p.similarity_score, 4),
                )
                for p in result.products
            ],
            total_found=result.total_found,
        )
    except Exception as e:
        logger.exception("Query processing error")
        raise HTTPException(status_code=500, detail=str(e))
    


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
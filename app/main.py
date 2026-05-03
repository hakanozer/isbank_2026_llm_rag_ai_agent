"""
AI Commerce Assistant — FastAPI giriş noktası.
Tüm route'lar, middleware ve lifecycle eventler buradan yönetilir.
"""
from fastapi import FastAPI  # Ana uygulama sınıfı
from fastapi.middleware.cors import CORSMiddleware  # CORS politikasını yönetir
from contextlib import asynccontextmanager  # Uygulama yaşam döngüsü yönetimi

from app.core.config import settings
from app.api.routes import router


@asynccontextmanager  # Asenkron bağlam yöneticisi
async def lifespan(app: FastAPI):
    """Uygulama başlarken ve kapanırken çalışacak kod."""
    # Başlangıç
    print(f"🚀 {settings.app_name} başlatılıyor...")
    print(f"   Ortam  : {settings.app_env}")
    print(f"   Debug  : {settings.debug}")
    print(f"   LLM    : {settings.ollama_model} @ {settings.ollama_base_url}")
    yield
    # Kapanış
    print("🛑 Uygulama kapatılıyor...")


app = FastAPI(  # FastAPI uygulama örneği
    title=settings.app_name,
    description="LLM + RAG + AI Agent ile akıllı e-ticaret asistanı",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
)

# CORS ayarları (geliştirme için tüm origin'lere izin)
app.add_middleware(  # Ara katman yazılımı ekler
    CORSMiddleware,
    allow_origins=["*"],  # Tüm kaynaklara izin ver (geliştirme)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API route'larını kaydet
app.include_router(router, prefix="/api/v1")  # Router'ı uygulamaya bağlar


@app.get("/", tags=["health"])  # GET endpoint tanımı
async def root() -> dict:
    """Uygulama kök endpoint'i — sağlık kontrolü."""
    return {
        "app": settings.app_name,
        "version": "0.1.0",
        "status": "running",
        "env": settings.app_env,
    }
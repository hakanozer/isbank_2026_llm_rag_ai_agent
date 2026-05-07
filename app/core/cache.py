"""
Redis tabanlı önbellekleme servisi.
LLM sorgu yanıtlarını TTL ile saklar.
"""
from __future__ import annotations

import hashlib  # Kriptografik özet fonksiyonları
import json  # JSON okuma/yazma işlemleri
import logging  # Uygulama loglama
from typing import Any  # Tip ipuçları için

import redis.asyncio as aioredis

from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_TTL = 3600  # 1 saat


class CacheService:
    """Async Redis cache servisi."""

    def __init__(self, url: str = settings.redis_url) -> None:
        self._redis: aioredis.Redis | None = None
        self._url = url

    async def connect(self) -> None:
        """Redis bağlantısını açar."""
        self._redis = aioredis.from_url(
            self._url,
            encoding="utf-8",
            decode_responses=True,
        )
        await self._redis.ping()
        logger.info("Redis bağlantısı kuruldu: %s", self._url)

    async def disconnect(self) -> None:
        """Redis bağlantısını kapatır."""
        if self._redis:
            await self._redis.aclose()
            logger.info("Redis bağlantısı kapatıldı: %s", self._url)
            self._redis = None
            
    def _make_key(self, prefix: str, data: str) -> str:
        """Veri için deterministik cache anahtarı üretir."""
        hash_val = hashlib.md5(data.encode()).hexdigest()[:16]
        return f"ai_commerce:{prefix}:{hash_val}"
    
    
    async def get(self, prefix: str, query: str) -> Any | None:
        """Cache'den değer getirir. Bulamazsa None döner."""
        if self._redis is None:
            return None
        key = self._make_key(prefix, query)
        try:
            raw = await self._redis.get(key)
            if raw:
                logger.debug("Cache HIT: %s", key)
                return json.loads(raw)
        except Exception as e:
            logger.warning("Cache get hatası: %s", e)
        return None
    
    
    async def set(
        self,
        prefix: str,
        query: str,
        value: Any,
        ttl: int = DEFAULT_TTL,
    ) -> None:
        """Cache'e değer kaydeder."""
        if self._redis is None:
            return
        key = self._make_key(prefix, query)
        try:
            await self._redis.setex(key, ttl, json.dumps(value, ensure_ascii=False))
            logger.debug("Cache SET: %s (TTL: %ds)", key, ttl)
        except Exception as e:
            logger.warning("Cache set hatası: %s", e)
            self._redis = None

    async def invalidate(self, prefix: str) -> int:
        """Belirli bir prefix'e ait tüm cache'i temizler."""
        if self._redis is None:
            return 0
        pattern = f"ai_commerce:{prefix}:*"
        keys = await self._redis.keys(pattern)
        if keys:
            return await self._redis.delete(*keys)
        return 0


cache = CacheService()
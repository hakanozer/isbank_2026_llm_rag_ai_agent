# Qdrant tabanlı ürün vector store
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from qdrant_client.http.exceptions import ResponseHandlingException
import httpx
import pandas as pd
from app.rag.embedder import embedding_service

logger = logging.getLogger(__name__)

QDRANT_COLLECTION = "products"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

class ProductQdrantStore:
    def __init__(self):
        # Initialize Qdrant client (no compatibility flag to avoid unexpected kw errors)
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection = QDRANT_COLLECTION

    def build(self, csv_path: str = "data/processed/products_for_embedding.csv"):
        logger.info("Qdrant index oluşturuluyor: %s", csv_path)
        df = pd.read_csv(csv_path, encoding="utf-8")
        texts = df["embedding_text"].tolist()
        # Qdrant expects numeric IDs or UUIDs — keep as integers
        product_ids = df["id"].astype(int).tolist()
        names = df["name"].tolist()
        prices = df["price"].tolist()
        categories = df.get("category", pd.Series([""] * len(df))).tolist()
        brands = df.get("brand", pd.Series([""] * len(df))).tolist()
        vectors = embedding_service.embed_batch(texts, show_progress=True)
        # Koleksiyon oluştur (bağlantı hatalarını yakala)
        try:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
            )
        except (ResponseHandlingException, httpx.ConnectError) as e:
            raise RuntimeError(
                "Qdrant sunucusuna bağlanılamıyor. Lütfen `docker-compose up -d qdrant` ile başlatın."
            ) from e
        # Noktaları ekle
        points = [
            PointStruct(
                id=product_ids[i],
                vector=vectors[i].tolist(),
                payload={
                    "name": names[i],
                    "price": prices[i],
                    "category": categories[i],
                    "brand": brands[i],
                },
            )
            for i in range(len(texts))
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        logger.info("Qdrant index hazır: %d vektör", len(points))

    def search(self, query_text: str, top_k: int = 5, score_threshold: float = 0.3):
        query_vector = embedding_service.embed_text(query_text)
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
        )
        results = []
        for hit in hits:
            meta = hit.payload.copy()
            meta["score"] = float(hit.score)
            results.append(meta)
        return results

    @property
    def total_products(self) -> int:
        info = self.client.get_collection(self.collection)
        return info.vectors_count

qdrant_store = ProductQdrantStore()

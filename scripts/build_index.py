"""
FAISS index oluşturma scripti.
Modül 3'te hazırlanan products_for_embedding.csv dosyasını kullanır.

Çalıştırma: python scripts/build_index.py
"""
import time  # Zaman ölçümleri
from app.rag.vector_store import vector_store

if __name__ == "__main__":
    start = time.time()
    vector_store.build("data/processed/products_for_embedding.csv")
    elapsed = time.time() - start
    print(f"\nIndex oluşturuldu: {elapsed:.1f} saniye")
    print(f"Toplam vektör: {vector_store.total_products}")

    # Hızlı test araması
    print("\nTest araması: 'terletmeyen spor ayakkabı'")
    results = vector_store.search("terletmeyen spor ayakkabı", top_k=3)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['name']} — {r['price']} TL")
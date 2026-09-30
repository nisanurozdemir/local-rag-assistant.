"""
ingest.py
documents/ klasorundeki .txt ve .docx dosyalarini okur, paragraflara
boler (chunking), her parca icin Foundry Local'in embedding modeliyle
bir vektor uretir ve SQLite veritabanina kaydeder.

Kullanim:
    python ingest.py

Yeni dokuman ekledidiginde veya mevcutlari degistirdiginde bu betigi
tekrar calistir; eski kayitlari silip yeniden olusturur.
"""

from pathlib import Path

from docx import Document
from foundry_local_sdk import Configuration, FoundryLocalManager

import db

DOCUMENTS_DIR = Path(__file__).parent / "documents"
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b-generic-cpu:1"

# Chunk'lar en fazla kac karakter olsun (paragraflari bu boyutu asarsa boler)
MAX_CHUNK_CHARS = 800


def read_docx(path: Path) -> str:
    """Bir .docx dosyasinin paragraflarini okuyup birlestirir."""
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_documents() -> list[tuple[str, str]]:
    """documents/ klasorundeki .txt ve .docx dosyalarini okur.
    Dondurur: [(dosya_adi, tam_metin), ...]
    """
    docs = []
    for path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        docs.append((path.name, text))
    for path in sorted(DOCUMENTS_DIR.glob("*.docx")):
        text = read_docx(path)
        docs.append((path.name, text))
    return docs


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Metni bos satirlara gore paragraflara boler; cok uzun paragraflari
    da max_chars sinirina gore ayirir. Cok kisa/bos parcalari atar.
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for para in raw_paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            words = para.split()
            current = []
            current_len = 0
            for word in words:
                current.append(word)
                current_len += len(word) + 1
                if current_len >= max_chars:
                    chunks.append(" ".join(current))
                    current = []
                    current_len = 0
            if current:
                chunks.append(" ".join(current))

    return chunks


def main():
    docs = load_documents()
    if not docs:
        print(f"Uyari: {DOCUMENTS_DIR} klasorunde .txt/.docx dosyasi bulunamadi.")
        return

    print(f"{len(docs)} dokuman bulundu. Parcalara ayriliyor...")

    all_chunks: list[tuple[str, str]] = []
    for filename, text in docs:
        pieces = chunk_text(text)
        for piece in pieces:
            all_chunks.append((filename, piece))
        print(f"  - {filename}: {len(pieces)} parca")

    if not all_chunks:
        print("Hic parca uretilemedi, dokumanlarin icerigini kontrol et.")
        return

    print(f"\nToplam {len(all_chunks)} parca. Embedding modeli yukleniyor...")

    config = Configuration(app_name="local_rag_assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model_variant(EMBEDDING_MODEL_ALIAS)
    embedding_model.download(
        lambda p: print(f"\r  Indiriliyor: %{p:.1f}", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    texts = [chunk for _source, chunk in all_chunks]
    print("Embedding'ler hesaplaniyor (tek seferde toplu istek)...")
    response = embedding_client.generate_embeddings(texts)
    embeddings = [item.embedding for item in response.data]

    print("Veritabanina kaydediliyor...")
    db.init_db()
    db.clear_chunks()
    for (source, content), embedding in zip(all_chunks, embeddings):
        db.insert_chunk(source, content, embedding)

    embedding_model.unload()

    print(f"\nTamamlandi. Veritabaninda {db.count_chunks()} parca var.")
    print("Simdi 'python main.py' ile soru sorabilirsin.")


if __name__ == "__main__":
    main()

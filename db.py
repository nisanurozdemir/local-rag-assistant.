"""
db.py
Küçük bir SQLite yardımcı modülü.
Dokümanları parçalara (chunk) ayırıp, her parçanın metnini, kaynağını
(hangi dosyadan geldiğini) ve embedding vektörünü (JSON olarak) saklar.
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "knowledge_base.db"


def get_connection():
    """Veritabanına bağlantı döndürür (yoksa dosyayı oluşturur)."""
    return sqlite3.connect(DB_PATH)


def init_db():
    """chunks tablosunu oluşturur (zaten varsa dokunmaz)."""
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def clear_chunks():
    """Yeniden ingest yapmadan önce eski kayıtları temizler."""
    conn = get_connection()
    conn.execute("DELETE FROM chunks")
    conn.commit()
    conn.close()


def insert_chunk(source: str, content: str, embedding: list[float]):
    """Tek bir parçayı (chunk) veritabanına ekler."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO chunks (source, content, embedding) VALUES (?, ?, ?)",
        (source, content, json.dumps(embedding)),
    )
    conn.commit()
    conn.close()


def fetch_all_chunks():
    """Tüm parçaları (id, source, content, embedding) olarak döndürür."""
    conn = get_connection()
    rows = conn.execute("SELECT id, source, content, embedding FROM chunks").fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "source": row[1],
            "content": row[2],
            "embedding": json.loads(row[3]),
        }
        for row in rows
    ]


def count_chunks() -> int:
    conn = get_connection()
    n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    return n

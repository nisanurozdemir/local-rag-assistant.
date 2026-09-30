"""
test.py
Sistemi onceden tanimlanmis sorularla test eder, sonuclari bir markdown
rapor dosyasina yazar. Manuel olarak "python main.py" ile tek tek soru
sormak yerine, tum test setini tek seferde calistirip sonuclari
degerlendirmek icin kullanilir.

Kullanim:
    python test.py
"""

import math
from datetime import datetime

from foundry_local_sdk import Configuration, FoundryLocalManager

import db

CHAT_MODEL_ALIAS = "qwen2.5-1.5b-instruct-generic-cpu:4"
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b-generic-cpu:1"
TOP_K = 3

SYSTEM_PROMPT_TEMPLATE = (
    "Sen, sadece sana verilen baglamdaki bilgiyi kullanarak soru "
    "cevaplayan bir asistansin. Baglamda soruyla ilgili bilgi VARSA, "
    "o bilgiye dayanarak cevap ver. Baglam sorunun konusuyla tamamen "
    "alakasizsa (baglam baska bir konudan bahsediyorsa), kendi genel "
    "bilgini kullanma; bu durumda sadece 'Bu bilgi belgede yok.' de."
)

USER_PROMPT_TEMPLATE = (
    "Baglam:\n{context}\n\n"
    "Soru: {query}"
)

# Test sorulari: (soru, beklenti_notu)
# beklenti_notu sadece rapor okunurken hatirlatma amaclidir, otomatik
# dogrulama yapmaz - cevaplari sen goz ile degerlendireceksin.
TEST_QUESTIONS = [
    ("Foundry Local nedir?", "Cevaplanabilir - foundry_local.txt"),
    ("Foundry Local'i macOS'a nasil kurarim?", "Cevaplanabilir - foundry_local.txt"),
    ("RAG uc adimdan olusuyor, bunlar nelerdir?", "Cevaplanabilir - rag_kavrami.txt"),
    ("Kosinus benzerligi ne ise yarar?", "Cevaplanabilir - embedding_nedir.txt"),
    ("SQLite'in avantajlari nelerdir?", "Cevaplanabilir - sqlite_temelleri.txt"),
    ("Bu raporun gelecek calismalar onerileri neler?", "Cevaplanabilir - docx raporu"),
    ("IoT protokollerinin guvenlik acisindan karsilastirmasi nasil?", "Cevaplanabilir - docx raporu"),
    ("Fransa'nin baskenti neresi?", "Cevaplanamaz - belgede yok, model 'bilmiyorum' demeli"),
    ("Python'da liste nasil olusturulur?", "Cevaplanamaz - belgede yok"),
    ("Dun hava nasildi?", "Cevaplanamaz - belgede yok"),
]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def find_relevant(query_embedding, chunks, top_k=TOP_K):
    scored = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def answer_question(query, embedding_client, chat_client, all_chunks):
    query_response = embedding_client.generate_embedding(query)
    query_embedding = query_response.data[0].embedding

    results = find_relevant(query_embedding, all_chunks, top_k=TOP_K)

    if not results or results[0][0] < 0.2:
        context = "(Ilgili bir belge bulunamadi.)"
        sources = []
    else:
        context = "\n".join(
            f"[Kaynak: {chunk['source']}] {chunk['content']}"
            for _score, chunk in results
        )
        sources = sorted({chunk["source"] for _score, chunk in results})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(context=context, query=query)},
    ]

    full_response = ""
    for chunk in chat_client.complete_streaming_chat(messages):
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if not content:
            continue
        full_response += content
        if len(full_response) > 400:
            full_response += " [...]"
            break
        if len(full_response) > 60:
            tail = full_response[-40:]
            if full_response.count(tail) > 2:
                full_response += " [...]"
                break

    top_score = results[0][0] if results else 0.0
    return full_response, sources, top_score


def main():
    n_chunks = db.count_chunks()
    if n_chunks == 0:
        print("Veritabani bos. Once 'python ingest.py' calistirmalisin.")
        return

    print(f"Veritabaninda {n_chunks} parca bulundu. Modeller yukleniyor...")

    config = Configuration(app_name="local_rag_test")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model_variant(EMBEDDING_MODEL_ALIAS)
    embedding_model.download(lambda p: None)
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    chat_model = manager.catalog.get_model_variant(CHAT_MODEL_ALIAS)
    chat_model.download(lambda p: None)
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    all_chunks = db.fetch_all_chunks()

    print(f"Modeller hazir. {len(TEST_QUESTIONS)} soru test edilecek...\n")

    lines = []
    lines.append(f"# Test Raporu - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\nVeritabaninda {n_chunks} parca var. TOP_K={TOP_K}.\n")

    for i, (question, expectation) in enumerate(TEST_QUESTIONS, start=1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] {question}")
        answer, sources, top_score = answer_question(
            question, embedding_client, chat_client, all_chunks
        )

        lines.append(f"## {i}. {question}")
        lines.append(f"**Beklenti:** {expectation}")
        lines.append(f"**En yuksek benzerlik skoru:** {top_score:.3f}")
        lines.append(f"**Bulunan kaynaklar:** {', '.join(sources) if sources else '(yok)'}")
        lines.append(f"**Cevap:** {answer}")
        lines.append("")

   

    report_path = "test_raporu.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nTamamlandi. Rapor '{report_path}' dosyasina yazildi.")


if __name__ == "__main__":
    main()

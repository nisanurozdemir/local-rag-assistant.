"""
main.py
Yerel RAG asistaninin ana uygulamasi.
"""

import math

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


def main():
    n_chunks = db.count_chunks()
    if n_chunks == 0:
        print("Veritabani bos. Once 'python ingest.py' calistirmalisin.")
        return

    print(f"Veritabaninda {n_chunks} parca bulundu. Modeller yukleniyor...")
    print()

    config = Configuration(app_name="local_rag_assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model_variant(EMBEDDING_MODEL_ALIAS)
    embedding_model.download(
        lambda p: print(f"\r  Embedding modeli indiriliyor: %{p:.1f}", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    chat_model = manager.catalog.get_model_variant(CHAT_MODEL_ALIAS)
    chat_model.download(
        lambda p: print(f"\r  Sohbet modeli indiriliyor: %{p:.1f}", end="", flush=True)
    )
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    all_chunks = db.fetch_all_chunks()

    print()
    print("Modeller hazir. Dokuman koleksiyonun hakkinda soru sorabilirsin.")
    print('Cikmak icin "quit" yaz.')
    print()

    while True:
        query = input("Soru: ").strip()
        if not query or query.lower() in ("quit", "cik", "exit"):
            break

        query_response = embedding_client.generate_embedding(query)
        query_embedding = query_response.data[0].embedding

        results = find_relevant(query_embedding, all_chunks, top_k=TOP_K)

        if not results or results[0][0] < 0.2:
            context = "(Ilgili bir belge bulunamadi.)"
        else:
            context = "\n".join(
                f"[Kaynak: {chunk['source']}] {chunk['content']}"
                for _score, chunk in results
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(context=context, query=query)},
        ]

        print("Cevap: ", end="", flush=True)
        full_response = ""
        for chunk in chat_client.complete_streaming_chat(messages):
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if not content:
                continue
            print(content, end="", flush=True)
            full_response += content
            if len(full_response) > 400:
                print(" [...]", end="", flush=True)
                break
            if len(full_response) > 60:
                tail = full_response[-40:]
                if full_response.count(tail) > 2:
                    print(" [...]", end="", flush=True)
                    break
        print()
        print()

    embedding_model.unload()
    chat_model.unload()
    print("Modeller kapatildi. Gorusuruz!")


if __name__ == "__main__":
    main()

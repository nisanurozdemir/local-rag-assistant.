"""
app.py
Yerel RAG asistaninin Streamlit web arayuzu.

Calistirmak icin:
    streamlit run app.py
"""

import math

import streamlit as st
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


@st.cache_resource(show_spinner="Modeller yukleniyor (ilk seferinde biraz surebilir)...")
def load_models():
    config = Configuration(app_name="local_rag_assistant_web")
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

    return embedding_client, chat_client


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

    return full_response, sources


st.set_page_config(page_title="Yerel RAG Asistani", page_icon="📚")
st.title("📚 Yerel RAG Asistani")
st.caption("Foundry Local ile tamamen cihazinda calisan doküman soru-cevap asistani.")

n_chunks = db.count_chunks()
if n_chunks == 0:
    st.error("Veritabani bos. Once terminalde 'python ingest.py' calistirmalisin.")
    st.stop()

st.caption(f"Veritabaninda {n_chunks} parca var.")

embedding_client, chat_client = load_models()
all_chunks = db.fetch_all_chunks()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Kaynaklar: " + ", ".join(msg["sources"]))

query = st.chat_input("Dokumanlar hakkinda bir soru sor...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Dusunuyor..."):
            answer, sources = answer_question(query, embedding_client, chat_client, all_chunks)
        st.markdown(answer)
        if sources:
            st.caption("Kaynaklar: " + ", ".join(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

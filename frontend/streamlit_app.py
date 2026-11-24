import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Kurumsal RAG Asistanı", page_icon="📄", layout="wide")

st.title("📄 Kaynak Atıflı Kurumsal Doküman Asistanı")
st.caption("FastAPI • ChromaDB • Gemini Embedding & LLM")

with st.sidebar:
    st.header("📁 Doküman Yönetimi")
    
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"API Aktif | Kayıtlı Parça: {health['indexed_chunks']}")
    except Exception:
        st.error("FastAPI sunucusuna ulaşılamıyor. Lütfen backend'i başlatın.")

    uploaded_file = st.file_uploader("Sisteme PDF Yükle", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Dokümanı İndeksle", use_container_width=True):
            with st.spinner("PDF parçalanıyor ve vektörleştiriliyor..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                res = requests.post(f"{API_URL}/upload", files=files)
                if res.status_code == 200:
                    st.success(res.json()["message"])
                    st.rerun()
                else:
                    st.error(res.json().get("detail", "Yükleme başarısız."))

    st.divider()
    top_k = st.slider("Referans Alınacak Parça Sayısı (Top-K)", min_value=2, max_value=8, value=4)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("🔍 Kullanılan Kaynak Metinleri (Kanıtlar)"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**{idx}. {src['source']} — Sayfa {src['page']}** *(Vektör Mesafesi: {src['distance']})*")
                    st.info(src["snippet"])

if prompt := st.chat_input("Yüklenen dokümanlar hakkında bir soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Dokümanlar taranıyor ve yanıt oluşturuluyor..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": prompt, "top_k": top_k},
                    timeout=120
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]

                    st.markdown(answer)
                    if sources:
                        with st.expander("🔍 Kullanılan Kaynak Metinleri (Kanıtlar)"):
                            for idx, src in enumerate(sources, 1):
                                st.markdown(f"**{idx}. {src['source']} — Sayfa {src['page']}** *(Vektör Mesafesi: {src['distance']})*")
                                st.info(src["snippet"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(response.json().get("detail", "Bir hata oluştu."))
            except Exception as e:
                st.error(f"Bağlantı hatası: {e}")
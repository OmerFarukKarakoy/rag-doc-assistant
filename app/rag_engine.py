import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb
import time

# 1. Ortam değişkenlerini ve istemcileri başlat (Her istek için max 20 sn zaman aşımı)
load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=20000)
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="enterprise_docs")

def get_query_embedding(query: str) -> list[float]:
    """Kullanıcı sorusunu Gemini gemini-embedding-001 ile vektöre çevirir."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
    )
    return response.embeddings[0].values

def ask_question(question: str, top_k: int = 4) -> dict:
    """
    Soruyu ChromaDB'de arar, en alakalı 'top_k' metin parçasını bulur
    ve LLM ile kaynak atıflı yanıt üretir.
    """
    if collection.count() == 0:
        return {
            "answer": "Sistemde henüz indekslenmiş bir doküman bulunmuyor. Lütfen önce bir PDF yükleyin.",
            "sources": []
        }

    # 1. Retrieval: Soruyu vektöre çevir ve en benzer parçaları getir
    query_emb = get_query_embedding(question)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(top_k, collection.count())
    )

    retrieved_docs = results["documents"][0]
    retrieved_metas = results["metadatas"][0]
    retrieved_distances = results["distances"][0]

    # 2. Augmentation: Bağlamı ve kaynak listesini hazırla
    context_blocks = []
    sources_list = []

    for idx, (text, meta, dist) in enumerate(zip(retrieved_docs, retrieved_metas, retrieved_distances)):
        source_name = meta.get("source", "Bilinmeyen Kaynak")
        page_num = meta.get("page", "?")
        
        context_blocks.append(
            f"[BELGE {idx+1} | Kaynak: {source_name} | Sayfa: {page_num}]\n{text}"
        )
        
        sources_list.append({
            "source": source_name,
            "page": page_num,
            "snippet": text.strip(),
            "distance": round(float(dist), 4)
        })

    context_text = "\n\n---\n\n".join(context_blocks)

    # 3. Generation: Hem Gemini hem Gemma modelleriyle uyumlu birleşik prompt
    system_instruction = """Sen kurumsal bir doküman ve bilgi bankası asistanısın.
Görevin, kullanıcıların sorularını YALNIZCA sana verilen 'BAĞLAM' içindeki metinlere dayanarak yanıtlamaktır.

Katı Kurallar:
1. Bağlamda yer almayan hiçbir bilgiyi, varsayımı veya genel kültür bilgisini cevaba ekleme.
2. Eğer sorunun cevabı bağlamda yoksa, kesinlikle uydurma ve şu yanıtı ver: "Bu bilgi yüklenen dokümanlarda yer almıyor."
3. Kullandığın her bilginin veya cümlenin sonuna mutlaka hangi belgeden ve hangi sayfadan alındığını parantez içinde yaz. Format: [Kaynak: dosya_adi.pdf, Sayfa: X]
4. Açık, profesyonel ve net bir Türkçe kullan."""

    full_prompt = f"""{system_instruction}

BAĞLAM:
{context_text}

KULLANICI SORUSU:
{question}
"""

    # Farklı sunucu havuzlarına sahip aktif modeller (Gemini yoğun olursa Gemma devreye girer)
    models_to_try = [
        "gemini-4-argon",
        "gemini-4-carbon",
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemma-4-31b-it",
        "gemma-4-26b-a4b-it"
    ]
    response = None
    last_error = None

    for attempt in range(2):
        for model_name in models_to_try:
            try:
                print(f"Deneniyor: {model_name}...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                    )
                )
                print(f"Başarılı: {model_name}")
                break
            except Exception as e:
                last_error = e
                print(f"Uyarı ({model_name} atlandı): {e}")
        
        if response is not None:
            break
        time.sleep(2)

    if response is None:
        raise RuntimeError(f"Yanıt üretilemedi: {last_error}")

    return {
        "answer": response.text,
        "sources": sources_list
    }

if __name__ == "__main__":
    test_soru = input("Yüklediğin PDF hakkında bir soru sor: ")
    sonuc = ask_question(test_soru)
    
    print("\n--- YANIT ---")
    print(sonuc["answer"])
    
    print("\n--- KULLANILAN KAYNAK PARÇALARI ---")
    for s in sonuc["sources"]:
        print(f"- {s['source']} (Sayfa {s['page']}) | Mesafe: {s['distance']}")
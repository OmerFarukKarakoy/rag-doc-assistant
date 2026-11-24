import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
import chromadb

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="enterprise_docs")

def get_embedding(text: str) -> list[float]:
    """Metni Gemini gemini-embedding-001 modeli ile vektöre çevirir."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return response.embeddings[0].values

def ingest_pdf(file_path: str) -> int:
    """
    PDF dosyasını sayfa sayfa okur, parçalara ayırır ve
    kaynak + sayfa numarası bilgisiyle ChromaDB'ye kaydeder.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

    file_name = os.path.basename(file_path)
    reader = PdfReader(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      
        chunk_overlap=150,  
        separators=["\n\n", "\n", ".", " ", ""]
    )

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    print(f"'{file_name}' işleniyor (Toplam {len(reader.pages)} sayfa)...")

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1  
        raw_text = page.extract_text()

        if not raw_text or not raw_text.strip():
            continue

        chunks = text_splitter.split_text(raw_text)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{file_name}_p{page_num}_c{chunk_idx}"
            
            ids.append(chunk_id)
            documents.append(chunk)
            embeddings.append(get_embedding(chunk))
            metadatas.append({
                "source": file_name,
                "page": page_num,
                "chunk_index": chunk_idx
            })

    if not documents:
        print("Uyarı: PDF içinden okunabilir metin çıkarılamadı (Görsel/taranmış PDF olabilir).")
        return 0

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Başarılı! Toplam {len(documents)} metin parçası ChromaDB'ye kaydedildi.")
    return len(documents)

if __name__ == "__main__":
    pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
    if pdf_files:
        test_path = os.path.join("data", pdf_files[0])
        ingest_pdf(test_path)
    else:
        print("Test için 'data' klasörünün içine herhangi bir .pdf dosyası koyup tekrar çalıştır.")
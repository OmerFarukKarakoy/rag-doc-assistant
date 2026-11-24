import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.ingestion import ingest_pdf, collection
from app.rag_engine import ask_question

app = FastAPI(
    title="Enterprise Citation RAG API",
    description="Kaynak atıflı kurumsal doküman asistanı REST API servisi",
    version="1.0.0"
)

UPLOAD_DIR = "./data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 4

@app.get("/health")
def health_check():
    """Sistemin ve vektör veritabanının durumunu kontrol eder."""
    return {
        "status": "active",
        "indexed_chunks": collection.count()
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """PDF dosyasını kaydeder ve ChromaDB'ye indeksler."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Sadece PDF dosyaları yüklenebilir.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunk_count = ingest_pdf(file_path)
        return {
            "filename": file.filename,
            "indexed_chunks": chunk_count,
            "message": f"'{file.filename}' başarıyla işlendi ve {chunk_count} parça veritabanına eklendi."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Doküman işleme hatası: {str(e)}")

@app.post("/ask")
def query_documents(request: QuestionRequest):
    """Kullanıcı sorusuna kaynak atıflı yanıt üretir."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")

    try:
        result = ask_question(request.question, top_k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sorgu hatası: {str(e)}")
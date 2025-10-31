from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel

from app.database import get_session
from app.services import (
    chunk_markdown_by_sections,
    convert_to_markdown,
    get_unique_categories,
    insert_document_embeddings,
    vector_search
)

app = FastAPI(title="Vectoria API")

@app.get("/")
def read_root():
    return {"message": "Vectoria API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/api/categories")
async def list_categories(session: AsyncSession = Depends(get_session)):
    categories = await get_unique_categories(session)
    return categories

@app.get("/api/embeddings/count")
async def count_embeddings(session: AsyncSession = Depends(get_session)):
    from sqlalchemy import select, func
    from app.database import Embedding
    
    result = await session.execute(select(func.count(Embedding.id)))
    count = result.scalar()
    return {"total_embeddings": count}

@app.post("/api/embeddings/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: Optional[str] = None,
    session: AsyncSession = Depends(get_session)
):
    supported = ['.pdf', '.docx', '.doc', '.xlsx', '.pptx', '.jpg', '.jpeg', '.png', '.html', '.txt']
    ext = '.' + file.filename.split('.')[-1].lower() if file.filename else ".bin"
    
    if ext not in supported:
        raise HTTPException(status_code=400, detail=f"Unsupported format")
    
    file_bytes = await file.read()
    
    metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
        "file_extension": ext
    }
    if category:
        metadata["category"] = category
    
    try:
        count = await insert_document_embeddings(
            session=session,
            file_bytes=file_bytes,
            filename=file.filename or "file",
            metadata=metadata
        )
        
        return {
            "message": "Document processed successfully",
            "filename": file.filename,
            "chunks_created": count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/api/file/markdown")
async def file_to_markdown(
    file: UploadFile = File(...),
    chunk: bool = False,
):
    supported = ['.pdf', '.docx', '.doc', '.xlsx', '.pptx', '.jpg', '.jpeg', '.png', '.html', '.txt']
    ext = '.' + file.filename.split('.')[-1].lower() if file.filename else ".bin"
    
    if ext not in supported:
        raise HTTPException(status_code=400, detail=f"Unsupported format")
    
    file_bytes = await file.read()
    
    metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
        "file_extension": ext
    }
    try:
        text = convert_to_markdown(
            file_bytes=file_bytes,
            filename=file.filename or "file"
        )

        
        chunks = chunk_markdown_by_sections(text, max_chunk_size=1000) if chunk else None
        
        
        
        return {
            "message": "Document processed successfully",
            "filename": file.filename,
            "metadata": metadata,
            "markdown": text,
            "chunks": chunks or None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search(
    query: str,
    limit: int = 5,
    category_filter: Optional[str] = None,
    session: AsyncSession = Depends(get_session)
):
    try:
        results = await vector_search(
            session=session,
            query=query,
            limit=limit,
            category_filter=category_filter
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

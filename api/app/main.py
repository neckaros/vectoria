import logging
from pathlib import Path

import httpx
from app.auth import verify_token
from app.logging_config import logger
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel
from app.tools import get_filename_from_headers
from app.database import get_session
from app.models import SearchResponse
from app.services import (
    chunk_markdown_by_sections,
    convert_to_markdown,
    get_unique_categories,
    insert_document_embeddings,
    vector_search
)
logger = logging.getLogger("app")

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
    parent_url: str,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = None,
    downloadurl: Optional[str] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    hash: Optional[str] = Query(None, description="Optionally provide a hash of the file to be inserted to check for duplicates (will return a 400 error if exists and override is set to false)"),
    override: bool = Query(False, description="Override existing embedding if existing document with same hash exist for the project"),
    category: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
    auth: dict = Depends(verify_token),
):
    """
    Upload and vectorize document with full metadata.
    Provide either a file upload or a URL to download.
    """
    
    project = auth["project"]

    logger.info(f"Upload request: project={project} file={file.filename if file else None}, url={url}")

    if hash:
        from app.services import check_for_existing_embedding
        exists = await check_for_existing_embedding(session, hash)
        if exists:
            raise HTTPException(status_code=204, detail="Document with the same hash already exists.")
    
    # Validate: need either file or URL
    if not file and not downloadurl:
        raise HTTPException(status_code=400, detail="Provide either a file or a URL (downloadurl) to ingest")
    
    # Check if file was actually uploaded (file.filename exists and file has content)
    has_file = file is not None and file.filename and file.size != 0
    
    if has_file:
        assert file is not None  # Help type checker understand
        # Handle file upload
        filename = file.filename
        file_bytes = await file.read()
        logger.info(f"Uploaded file: {filename}, size: {len(file_bytes)} bytes")
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
    
    elif downloadurl:
        # Handle URL download
        try:
            logger.info(f"Downloading from URL: {url}")
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                response = await client.get(downloadurl)
                response.raise_for_status()
                
                file_bytes = response.content
                filename = get_filename_from_headers(response.headers, downloadurl)
                
                logger.info(f"Downloaded {len(file_bytes)} bytes, filename: {filename}")
                
        except httpx.HTTPError as e:
            logger.error(f"Failed to download URL: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download URL: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="No valid file or URL provided")
    if not filename:
        raise HTTPException(status_code=400, detail="Could not determine filename")
    # Validate file extension
    supported = ['.pdf', '.docx', '.doc', '.xlsx', '.pptx', '.jpg', '.jpeg', '.png', '.html', '.txt', '.md']
    ext = Path(filename).suffix.lower()
    
    if ext not in supported:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported format: {ext}. Supported: {', '.join(supported)}"
        )
    
    # Process the document
    try:
        count = await insert_document_embeddings(
            session=session,
            project=project,
            file_bytes=file_bytes,
            filename=filename,
            title=title or filename,
            author=author,
            category=category,
            hash=hash,
            url=url,
            parent_url=parent_url
        )
        
        return {
            "message": "Document processed successfully",
            "filename": filename,
            "chunks_created": count,
            "source": "url" if url and not has_file else "upload"
        }
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
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

@app.post("/api/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., description="Search query text"),
    limit: int = Query(5, description="Maximum number of final results to return"),
    initial_k: int = Query(20, description="Number of candidates to retrieve before reranking"),
    category_filter: Optional[str] = Query(None, description="Filter results by category"),
    source_filter: Optional[str] = Query(None, description="Filter results by source"),
    author_filter: Optional[str] = Query(None, description="Filter results by author"),
    parent_url: Optional[str] = Query(None, description="Filter results by parent_url (use %% at the end for prefix match)"),
    rerank: bool = Query(False, description="Enable cross-encoder reranking for better accuracy (slower)"),
    session: AsyncSession = Depends(get_session),
    auth: dict = Depends(verify_token),
):
    """
    Vector similarity search with optional reranking.
    
    Retrieves documents most similar to the query using vector embeddings,
    with optional reranking to improve relevance.
    """
    project = auth["project"]
    logger.info(f"Search: query='{query}', rerank={rerank}")
    
    try:
        results = await vector_search(
            session=session,
            query=query,
            limit=limit,
            category_filter=category_filter,
            author_filter=author_filter,
            source_filter=source_filter,
            parent_url=parent_url,
            rerank=rerank,
            initial_k=initial_k
        )
        
        return SearchResponse(
            query=query,
            results=results,
            count=len(results),
            reranked=rerank
        )
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
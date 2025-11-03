import hashlib
import io
import logging
from typing import List, Dict, Any, Optional, Union
from markitdown import MarkItDown
from sqlalchemy import Integer, String, and_, bindparam, select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from openai import OpenAI
from app.database import Embedding
from app.config import OPENAI_API_KEY
from app.models import EmbeddingResponse

client = OpenAI(api_key=OPENAI_API_KEY)
markitdown = MarkItDown()
logger = logging.getLogger("services")

def compute_hash(data: Union[str, bytes]) -> str:
    """
    Compute SHA256 hash of content for deduplication.
    Accepts both string and bytes input.
    """
    if isinstance(data, str):
        data = data.encode()
    
    return hashlib.sha256(data).hexdigest()

def convert_to_markdown(file_bytes: bytes, filename: str) -> str:
    """Convert any supported document to Markdown"""
    file_obj = io.BytesIO(file_bytes)
    result = markitdown.convert_stream(file_obj, file_extension=filename.split('.')[-1])
    return result.text_content


def chunk_markdown_by_sections(
    markdown_text: str, 
    max_chunk_size: int = 1000,
    min_chunk_size: int = 200,
    overlap_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Split markdown by headers while respecting size constraints and overlap.
    
    Args:
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters (chunks below this get merged with previous)
        overlap_size: Number of characters to overlap between chunks
    """
    chunks = []
    lines = markdown_text.split('\n')
    
    current_chunk = []
    current_header = ""
    current_level = 0
    chunk_size = 0
    overlap_buffer = []  # Store lines for overlap
    
    def save_chunk():
        """Helper to save current chunk if it meets minimum size"""
        nonlocal current_chunk, chunks, overlap_buffer
        
        if not current_chunk:
            return
            
        content = '\n'.join(current_chunk).strip()
        size = len(content)
        
        # If chunk is too small, merge with previous chunk
        if size < min_chunk_size and chunks:
            chunks[-1]['content'] += '\n\n' + content
            chunks[-1]['size'] = len(chunks[-1]['content'])
        elif size > 0:  # Only save non-empty chunks
            chunks.append({
                'content': content,
                'header': current_header,
                'level': current_level,
                'size': size
            })
        
        # Keep last N characters for overlap
        if overlap_size > 0 and current_chunk:
            overlap_text = '\n'.join(current_chunk)[-overlap_size:]
            overlap_buffer = overlap_text.split('\n')
    
    for line in lines:
        line_size = len(line)
        
        # Detect markdown headers
        if line.startswith('#'):
            header_level = len(line) - len(line.lstrip('#'))
            
            # Start new chunk on major headers (h1, h2) or when exceeding max size
            if (header_level <= 2 or chunk_size > max_chunk_size) and current_chunk:
                save_chunk()
                
                # Start new chunk with overlap from previous
                current_chunk = overlap_buffer.copy()
                chunk_size = sum(len(l) for l in overlap_buffer)
                overlap_buffer = []
            
            current_header = line.lstrip('#').strip()
            current_level = header_level
        
        current_chunk.append(line)
        chunk_size += line_size
        
        # Force split if chunk too large (with some margin)
        if chunk_size > max_chunk_size * 1.5:
            save_chunk()
            
            # Start new chunk with overlap
            current_chunk = overlap_buffer.copy()
            chunk_size = sum(len(l) for l in overlap_buffer)
            overlap_buffer = []
    
    # Save remaining content
    if current_chunk:
        save_chunk()
    
    return [c for c in chunks if c['content']]



def get_embedding(text: str) -> List[float]:
    """Get OpenAI embedding for text"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


async def insert_document_embeddings(
    session: AsyncSession,
    file_bytes: bytes,
    filename: str,
    hash: Optional[str] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    category: Optional[str] = None,
    url: Optional[str] = None,
    parent_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> int:
    
    if not hash:
        hash = compute_hash(file_bytes)


    existing = await session.execute(
        select(Embedding).where(Embedding.hash == hash)
    )
    if existing.scalars().first():
        return 0


    """Insert embeddings with full document metadata"""
    markdown_text = convert_to_markdown(file_bytes, filename) if filename != ".md" else file_bytes.decode()
    chunks = chunk_markdown_by_sections(markdown_text)
    
    count = 0
    for i, chunk_data in enumerate(chunks):
        # Check if this chunk already exists (deduplication)
        
      
        embedding_vector = get_embedding(chunk_data['content'])
        
        embedding_record = Embedding(
            title=title or filename,
            author=author,
            mimetype=get_mimetype(filename),  # Helper function
            category=category,
            source=filename,
            url=url,
            parent_url=parent_url,
            chunk_index=i,
            total_chunks=len(chunks),
            header=chunk_data['header'],
            header_level=chunk_data['level'],
            markdown=chunk_data['content'],
            hash=hash,
            embedding=embedding_vector,
            doc_metadata=metadata
        )
        
        session.add(embedding_record)
        count += 1
    
    await session.commit()
    return count


def get_mimetype(filename: str) -> str:
    """Get MIME type from filename"""
    ext_to_mime = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.txt': 'text/plain',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.html': 'text/html',
    }
    ext = '.' + filename.split('.')[-1].lower()
    return ext_to_mime.get(ext, 'application/octet-stream')


async def get_unique_categories(session: AsyncSession) -> List[str]:
    """Get all unique categories from metadata"""
    query = select(
        func.distinct(
            func.jsonb_extract_path_text(Embedding.doc_metadata, 'category')
        ).label('category')
    ).where(
        Embedding.doc_metadata['category'].isnot(None)
    )
    
    result = await session.execute(query)
    categories = [row.category for row in result if row.category]
    return categories


from sqlalchemy import select, and_, true

async def vector_search(
    session: AsyncSession,
    query: str,
    limit: int = 5,
    category_filter: Optional[str] = None,
    source_filter: Optional[str] = None,
    author_filter: Optional[str] = None,
    parent_url: Optional[str] = None,
    rerank: bool = False,
    initial_k: int = 20
) -> List[EmbeddingResponse]:
    """Perform vector similarity search"""
    logger.info(f"Vector search: '{query[:50]}...', rerank={rerank}")
    if initial_k <= limit:
        initial_k = limit * 5
    query_embedding = get_embedding(query)
    retrieve_limit = initial_k if rerank else limit

    if parent_url and parent_url.endswith("%"):
        start_parent_url = parent_url
        strict_parent_url = None
    elif parent_url:
        strict_parent_url = parent_url
        start_parent_url = None
    else:
        strict_parent_url = None
        start_parent_url = None
    sql = text("""
        SELECT 
            id, title, author, mimetype, category, source, url, parent_url,
            chunk_index, total_chunks, header, header_level, markdown, hash,
            metadata, created_at, last_modified,
            1 - (embedding <=> CAST(:query_embedding AS vector)) as similarity
        FROM embeddings
        WHERE 
            (:category_filter IS NULL OR category = :category_filter)
            AND (:author_filter IS NULL OR author = :author_filter)
            AND (:source_filter IS NULL OR source = :source_filter)
            AND (:start_parent_url IS NULL OR parent_url LIKE :start_parent_url)
            AND (:strict_parent_url IS NULL OR parent_url = :strict_parent_url)
        ORDER BY embedding <=> CAST(:query_embedding AS vector)
        LIMIT :retrieve_limit
    """).bindparams(
        bindparam("query_embedding", value=str(query_embedding), type_=String),  # SQLAlchemy String
        bindparam("category_filter", value=category_filter, type_=String),
        bindparam("author_filter", value=author_filter, type_=String),
        bindparam("source_filter", value=source_filter, type_=String),
        bindparam("start_parent_url", value=start_parent_url, type_=String),
        bindparam("strict_parent_url", value=strict_parent_url, type_=String),
        bindparam("retrieve_limit", value=retrieve_limit, type_=Integer),  # SQLAlchemy Integer
    )
    
    result = await session.execute(
        sql,
        {
            "query_embedding": str(query_embedding),
            "category_filter": category_filter,
            "author_filter": author_filter,
            "retrieve_limit": retrieve_limit
        }
    )
    
    
    # Convert to Pydantic
    results = [
        EmbeddingResponse.model_validate({
            "id": row.id,
            "title": row.title,
            "author": row.author,
            "mimetype": row.mimetype,
            "category": row.category,
            "source": row.source,
            "url": row.url,
            "parent_url": row.parent_url,
            "chunk_index": row.chunk_index,
            "total_chunks": row.total_chunks,
            "header": row.header,
            "header_level": row.header_level,
            "markdown": row.markdown,
            "hash": row.hash,
            "metadata": row.metadata,
            "created_at": row.created_at,
            "last_modified": row.last_modified,
            "vector_similarity": float(row.similarity),
        })
        for row in result
    ]
    
    logger.info(f"Retrieved {len(results)} candidates")
    
    if rerank and len(results) > limit:
        results = await rerank_results(query, results, top_k=limit)
    
    return results[:limit]


async def rerank_results(
    query: str,
    documents: List[EmbeddingResponse],
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> List[EmbeddingResponse]:
    """Rerank with Pydantic models"""
    from sentence_transformers import CrossEncoder
    
    try:
        logger.debug(f"Loading reranker: {model_name}")
        reranker = CrossEncoder(model_name, max_length=512)
        
        pairs = [
            [query[:200], doc.content[:300]]  # Use .content (typed!)
            for doc in documents
        ]
        
        scores = reranker.predict(pairs)
        
        # Update Pydantic models with rerank scores
        for i, doc in enumerate(documents):
            doc.rerank_score = float(scores[i])
            doc.original_rank = i + 1
        
        # Sort by rerank score
        reranked = sorted(documents, key=lambda x: x.rerank_score or 0, reverse=True)
        
        logger.info(f"Reranking complete. Top score: {reranked[0].rerank_score:.3f}")
        return reranked[:top_k]
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return documents[:top_k]


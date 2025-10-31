import hashlib
import io
from typing import List, Dict, Any, Optional, Union
from markitdown import MarkItDown
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from openai import OpenAI
from app.database import Embedding
from app.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
markitdown = MarkItDown()

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
    markdown_text = convert_to_markdown(file_bytes, filename)
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

async def vector_search(
    session: AsyncSession,
    query: str,
    limit: int = 5,
    category_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search
    """
    # Get query embedding
    query_embedding = get_embedding(query)
    
    # Build SQL query with cosine distance
    sql = text("""
        SELECT 
            id,
            source_url,
            content,
            metadata,
            created_at,
            1 - (embedding <=> :query_embedding) as similarity
        FROM embeddings
        WHERE (:category_filter IS NULL OR metadata->>'category' = :category_filter)
        ORDER BY embedding <=> :query_embedding
        LIMIT :limit
    """)
    
    result = await session.execute(
        sql,
        {
            "query_embedding": str(query_embedding),
            "limit": limit,
            "category_filter": category_filter
        }
    )
    
    results = []
    for row in result:
        results.append({
            "id": row.id,
            "source_url": row.source_url,
            "content": row.content,
            "metadata": row.metadata,
            "created_at": row.created_at,
            "similarity": float(row.similarity)
        })
    
    return results

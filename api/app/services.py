import io
from typing import List, Dict, Any, Optional
from markitdown import MarkItDown
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from openai import OpenAI
from app.database import Embedding
from app.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
markitdown = MarkItDown()


def convert_to_markdown(file_bytes: bytes, filename: str) -> str:
    """Convert any supported document to Markdown"""
    file_obj = io.BytesIO(file_bytes)
    result = markitdown.convert_stream(file_obj, file_extension=filename.split('.')[-1])
    return result.text_content


def chunk_markdown_by_sections(markdown_text: str, max_chunk_size: int = 1000) -> List[Dict[str, Any]]:
    """
    Split markdown by headers while respecting max size
    Returns list of dicts with content and metadata
    """
    chunks = []
    lines = markdown_text.split('\n')
    
    current_chunk = []
    current_header = ""
    current_level = 0
    chunk_size = 0
    
    for line in lines:
        line_size = len(line)
        
        # Detect markdown headers
        if line.startswith('#'):
            header_level = len(line) - len(line.lstrip('#'))
            current_header = line.lstrip('#').strip()
            current_level = header_level
            
            # Start new chunk if we have content and hit a major header
            if current_chunk and (header_level <= 2 or chunk_size > max_chunk_size):
                chunks.append({
                    'content': '\n'.join(current_chunk).strip(),
                    'header': current_header,
                    'level': current_level,
                    'size': chunk_size
                })
                current_chunk = []
                chunk_size = 0
        
        current_chunk.append(line)
        chunk_size += line_size
        
        # Force split if chunk too large
        if chunk_size > max_chunk_size * 1.5:
            chunks.append({
                'content': '\n'.join(current_chunk).strip(),
                'header': current_header,
                'level': current_level,
                'size': chunk_size
            })
            current_chunk = []
            chunk_size = 0
    
    # Add remaining content
    if current_chunk:
        chunks.append({
            'content': '\n'.join(current_chunk).strip(),
            'header': current_header,
            'level': current_level,
            'size': chunk_size
        })
    
    return [c for c in chunks if c['content']]  # Filter empty chunks


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
    metadata: Optional[Dict[str, Any]] = None
) -> int:
    """
    Convert document to Markdown, chunk intelligently, and insert embeddings
    """
    # Convert to markdown
    markdown_text = convert_to_markdown(file_bytes, filename)
    
    # Chunk by structure
    chunks = chunk_markdown_by_sections(markdown_text, max_chunk_size=1000)
    
    # Create and insert embeddings
    count = 0
    for i, chunk_data in enumerate(chunks):
        # Get embedding
        embedding_vector = get_embedding(chunk_data['content'])
        
        # Prepare metadata
        chunk_metadata = metadata.copy() if metadata else {}
        chunk_metadata.update({
            "chunk_index": i,
            "total_chunks": len(chunks),
            "header": chunk_data['header'],
            "header_level": chunk_data['level'],
            "text_length": chunk_data['size'],
            "format": "markdown"
        })
        
        # Create embedding record
        embedding_record = Embedding(
            source_url=filename,
            content=chunk_data['content'],
            doc_metadata=chunk_metadata,
            embedding=embedding_vector
        )
        
        session.add(embedding_record)
        count += 1
    
    await session.commit()
    return count

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

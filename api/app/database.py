from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Index, Integer, String, Text, DateTime, text
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from datetime import datetime
from app.config import DATABASE_URL

# Convert postgresql:// to postgresql+asyncpg://
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
print(f"database: {ASYNC_DATABASE_URL}")

engine = create_async_engine(ASYNC_DATABASE_URL, echo=True)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

class Embedding(Base):
    __tablename__ = "embeddings"
    
    # Primary key
    id = Column(Integer, primary_key=True)
    
    # Document metadata
    title = Column(Text, nullable=False, index=True)
    author = Column(Text, index=True)
    mimetype = Column(String(100))  # e.g. "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    category = Column(String(100), index=True)
    
    # Source tracking
    source = Column(Text, nullable=False, index=True)  # Original filename/path
    url = Column(Text)  # If from web
    parent_url = Column(Text)  # Referrer URL
    
    # Chunk metadata
    chunk_index = Column(Integer, nullable=False)
    total_chunks = Column(Integer, nullable=False)
    header = Column(Text)  # Section heading
    header_level = Column(Integer)  # Markdown level (1-6)
    
    # Content
    markdown = Column(Text, nullable=False)  # The actual chunk text
    hash = Column(String(64), nullable=False, unique=True, index=True)  # SHA256 of content for dedup
    
    # Vector embedding
    embedding = Column(Vector(1536), nullable=False)
    
    # Additional metadata (dynamic fields)
    doc_metadata = Column("metadata", JSONB)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    last_modified = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session

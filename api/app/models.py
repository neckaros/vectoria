from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from app.database import Embedding


class ContentItem(BaseModel):
    """Generic content item for webpages or files.

    key: primary key string identifying the item; use the page URL for web pages
         or a scheme-qualified path for files e.g. "sharepoint://{path-or-id}".
    """

    key: str
    title: str = "" # name of the webpage, filename of the document...
    markdown: str 
    last_modified: Optional[str] = None # last mod ISO 8601 date string of the page or document
    author: Optional[str] = None
    mimetype: Optional[str] = None
    category: Optional[str] = None # aribtrary category string for filtering
    source: Optional[str] = None # kind of ingestion ex. "sharepoint", "website"
    url: Optional[str] = None # url of the parsed document (include scheme: https:// for web, sharepoint:// for sharepoint...)
    parent_url: Optional[str] = None # for website url of the intially crawled page for sharepoint, drive, filesystem: the parent folder with the scheme
    doc_hash: Optional[str] = None # sha1 hash of the markdown content for deduplication



class EmbeddingBase(BaseModel):
    """Base embedding schema"""
    model_config = ConfigDict(
        from_attributes=True,
        # Allow fields that aren't in SQLAlchemy
    )
    
    id: int
    title: str
    author: Optional[str] = None
    mimetype: Optional[str] = None
    category: Optional[str] = None
    source: str
    url: Optional[str] = None
    parent_url: Optional[str] = None
    chunk_index: int
    total_chunks: int
    header: Optional[str] = None
    header_level: Optional[int] = None
    content: str = Field(..., alias="markdown")
    hash: str
    metadata: Optional[dict] = None
    created_at: datetime
    last_modified: datetime


class EmbeddingResponse(EmbeddingBase):
    """Response schema for search results (adds search-specific fields)"""
    vector_similarity: float
    rerank_score: Optional[float] = None
    original_rank: Optional[int] = None

class SearchResponse(BaseModel):
    """Search API response"""
    query: str
    results: list[EmbeddingResponse]
    count: int
    reranked: bool
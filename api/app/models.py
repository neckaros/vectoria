from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


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



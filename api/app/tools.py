import logging
import re
import httpx
from pathlib import Path
from urllib.parse import unquote, urlparse

logger = logging.getLogger("tools")

def get_filename_from_headers(headers: httpx.Headers, url: str) -> str:
    """
    Extract filename from HTTP headers or URL.
    Tries Content-Disposition first, then falls back to URL path.
    """
    # Try Content-Disposition header first
    content_disposition = headers.get('content-disposition')
    
    if content_disposition:
        # Try RFC 5987 encoded filename (filename*=UTF-8''...)
        match = re.search(r"filename\*=(?:UTF-8'')?([^;\r\n]+)", content_disposition, re.IGNORECASE)
        if match:
            filename = unquote(match.group(1).strip("\"'"))
            logger.debug(f"Extracted filename from Content-Disposition (RFC 5987): {filename}")
            return filename
        
        # Try standard filename parameter (filename="..." or filename=...)
        match = re.search(r'filename="?([^";\r\n]+)"?', content_disposition, re.IGNORECASE)
        if match:
            filename = match.group(1).strip("\"'")
            logger.debug(f"Extracted filename from Content-Disposition: {filename}")
            return filename
    
    # Fallback: extract from URL path
    parsed_url = urlparse(url)
    path = parsed_url.path
    
    if path and path != '/':
        filename = Path(path).name
        if filename:
            # URL decode the filename
            filename = unquote(filename)
            logger.debug(f"Extracted filename from URL path: {filename}")
            return filename
    
    # Last resort: use domain + timestamp
    domain = parsed_url.netloc or "download"
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{domain}_{timestamp}.pdf"  # Assume PDF if unknown
    logger.warning(f"Could not determine filename, using generated: {filename}")
    
    return filename
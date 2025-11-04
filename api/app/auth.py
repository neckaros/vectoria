import json
import logging
import os
from typing import Optional, Dict, List
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger("app")

security = HTTPBearer()

def get_valid_tokens() -> Dict[str, List[str]]:
    """Load valid tokens from environment"""
    tokens_json = os.getenv("API_TOKENS", "{}")
    logger.info(f"Raw API_TOKENS env: {repr(tokens_json)}")
    try:
        tokens = json.loads(tokens_json)
        logger.info(f"Loaded tokens for projects: {list(tokens.keys())}")
        return tokens
    except json.JSONDecodeError:
        logger.error("Invalid JSON in API_TOKENS environment variable")
        return {}

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Verify Bearer token from Authorization header"""
    token = credentials.credentials
    valid_tokens = get_valid_tokens()
    
    # Check if token exists in any project
    for project, token_list in valid_tokens.items():
        logger.info(f"[verify_token] Checking project '{project}' with {len(token_list)} token(s)")
        
        for idx, stored_token in enumerate(token_list):
            logger.info(f"[verify_token]   Comparing token {idx}:")
            logger.info(f"[verify_token]     Incoming: '{token}'")
            logger.info(f"[verify_token]     Stored:   '{stored_token}'")
            logger.info(f"[verify_token]     Equal: {token == stored_token}")
            
            if token == stored_token:
                logger.info(f"[verify_token] ✓ Token verified for project: {project}")
                return {"project": project, "token": token}
    
    logger.warning(f"Invalid token attempted: {token[:10]}...")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )

#!/usr/bin/env python
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv("../.env")

def main():
    """Run the FastAPI development server"""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7980,
        reload=True
    )

if __name__ == "__main__":
    main()
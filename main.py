"""
URL Summarizer API - Extract and summarize content from any URL using AI
Works with Ollama (local) or OpenAI (cloud)
"""
import os
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx
import trafilatura


# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "minimax-m2.5:cloud")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_OPENAI = bool(OPENAI_API_KEY)


class SummarizeRequest(BaseModel):
    url: str
    prompt: str = "Summarize this article in 3 bullet points"
    model: Optional[str] = None


class SummarizeResponse(BaseModel):
    url: str
    summary: str
    model: str
    success: bool = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 URL Summarizer starting...")
    print(f"📡 Ollama: {OLLAMA_BASE_URL}")
    print(f"🤖 Default model: {DEFAULT_MODEL}")
    print(f"🔑 Using: {'OpenAI' if USE_OPENAI else 'Ollama'}")
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title="URL Summarizer API",
    description="AI-powered URL content extraction and summarization",
    version="1.0.0",
    lifespan=lifespan
)


def extract_content(url: str) -> str:
    """Extract main content from URL"""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise ValueError("Failed to fetch URL")
        
        text = trafilatura.extract(downloaded, include_tables=False, include_images=False)
        if not text:
            raise ValueError("No content extracted")
        
        return text
    except Exception as e:
        raise ValueError(f"Content extraction failed: {str(e)}")


async def summarize_with_ollama(content: str, prompt: str, model: str) -> str:
    """Summarize using local Ollama"""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            full_prompt = f"""{prompt}

Content:
{content[:8000]}

Summary:"""

            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                }
            )
            
            if response.status_code != 200:
                raise ValueError(f"Ollama error")
            
            return response.json().get("response", "").strip()
            
    except Exception as e:
        raise ValueError(f"Ollama failed: {str(e)}")


async def summarize_with_openai(content: str, prompt: str, model: str) -> str:
    """Summarize using OpenAI API"""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model or "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that summarizes content clearly."},
                        {"role": "user", "content": f"{prompt}\n\nContent:\n{content[:10000]}"}
                    ],
                    "temperature": 0.3
                }
            )
            
            if response.status_code != 200:
                raise ValueError(f"OpenAI error: {response.text}")
            
            return response.json()["choices"][0]["message"]["content"]
            
    except Exception as e:
        raise ValueError(f"OpenAI failed: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve web interface"""
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>URL Summarizer</h1><p>API running. Use POST /summarize</p>"


@app.get("/health")
async def health():
    return {
        "service": "URL Summarizer API",
        "version": "1.0.0",
        "ai_provider": "openai" if USE_OPENAI else "ollama",
        "model": DEFAULT_MODEL
    }


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Summarize a URL"""
    model = request.model or DEFAULT_MODEL
    
    try:
        # Extract content
        content = extract_content(request.url)
        
        # Summarize
        if USE_OPENAI:
            summary = await summarize_with_openai(content, request.prompt, model)
        else:
            summary = await summarize_with_ollama(content, request.prompt, model)
        
        return SummarizeResponse(
            url=request.url,
            summary=summary,
            model=model
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

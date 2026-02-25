# URL Summarizer 🧠

AI-powered tool that extracts and summarizes content from any URL using AI.

## Features

- 📄 Extract content from any webpage
- 🤖 AI summarization (Ollama or OpenAI)
- 🎨 Beautiful web interface
- ⚡ Fast and lightweight

## Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python main.py
```

Then open http://localhost:8000

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama API URL (local) | http://127.0.0.1:11434 |
| `DEFAULT_MODEL` | AI model to use | llama3.2 |
| `OPENAI_API_KEY` | OpenAI API key (for cloud) | - |

## Deploy to Render (Free)

1. **Push to GitHub:**
   - Create a new repository on GitHub
   - Push this code:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/url-summarizer.git
   git branch -M main
   git push -u origin main
   ```

2. **Deploy on Render:**
   - Go to https://dashboard.render.com
   - Create new "Web Service"
   - Connect your GitHub repository
   - Set environment variables:
     - `OPENAI_API_KEY` = your OpenAI key (get free at https://platform.openai.com/api-keys)
   - Build command: `pip install -r requirements.txt`
   - Start command: `python main.py`

3. **Done!** You'll get a free URL like `https://your-app.onrender.com`

## API Usage

```bash
curl -X POST https://your-app.onrender.com/summarize \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "prompt": "Summarize in 3 bullet points"}'
```

## Tech Stack

- FastAPI (Python web framework)
- Trafilatura (content extraction)
- Ollama / OpenAI (AI summarization)
- Beautiful UI (vanilla HTML/CSS)

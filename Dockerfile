FROM python:3.12-slim

WORKDIR /app

# Install uv for fast package management
RUN pip install uv

# Copy requirements first for better caching
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy app files
COPY . .

# Expose port
EXPOSE 8000

# Run the app
CMD ["python", "main.py"]

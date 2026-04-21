FROM python:3.11-slim

# Non-root user for HuggingFace Spaces compatibility
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Switch to non-root
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# server.app:app — runs server/app.py from /app working directory
# models.py, client.py, inference.py live at /app root (on PYTHONPATH automatically)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

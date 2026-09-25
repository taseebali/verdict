# --- Build the React frontend ---
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Python runtime serving API + built frontend ---
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/mpl \
    NUMBA_CACHE_DIR=/tmp/numba \
    PORT=7860

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py pyproject.toml ./
COPY src ./src
COPY backend/app ./backend/app
COPY data/verdict_demo.csv ./data/verdict_demo.csv
COPY --from=frontend /frontend/dist ./frontend/dist

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd --uid 1000 --no-create-home verdict
USER 1000

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '7860') + '/api/health')"

CMD ["python", "app.py"]

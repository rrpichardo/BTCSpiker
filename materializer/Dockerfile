FROM python:3.11-slim

WORKDIR /app

COPY materializer/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY materializer/ materializer/

# materializer.py does flat imports (no package __init__.py) — mirrors the
# PYTHONPATH pattern in docker/Dockerfile.worker.
ENV PYTHONPATH=/app/materializer
ENV PYTHONUNBUFFERED=1

EXPOSE 8090

CMD ["uvicorn", "materializer:app", "--host", "0.0.0.0", "--port", "8090"]

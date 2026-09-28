FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app

# For real inference, install requirements-ml.txt, provide weights, and set DETECTOR_BACKEND=yolo.
ENV DETECTOR_BACKEND=demo \
    REPORT_BACKEND=auto

USER nobody
EXPOSE 8000
# Hosts such as Render inject PORT; default to 8000 elsewhere.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

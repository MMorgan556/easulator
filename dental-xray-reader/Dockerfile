FROM python:3.11-slim

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app

# For real inference, build from requirements-ml.txt, mount weights, and set DETECTOR_BACKEND=yolo.
ENV DETECTOR_BACKEND=demo \
    REPORT_BACKEND=auto

EXPOSE 8000
USER nobody
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Même version de Python que pour l'entraînement
FROM python:3.13-slim
WORKDIR /app
# Dépendances d'abord : cette couche reste en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py features.py ./
COPY model/ model/
COPY static/ static/
# Pas d'exécution en root
RUN useradd --create-home appuser && mkdir -p logs && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

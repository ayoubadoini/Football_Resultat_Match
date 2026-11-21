# football-api
```
pip install -r requirements.txt
python -m uvicorn app:app --port 8000      # puis http://127.0.0.1:8000  (docs : /docs)
python -m pytest -q                         # tests
docker build -t football-api:v1 . && docker run -p 8000:8000 football-api:v1
``
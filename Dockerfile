FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# web/dist is served at / if it was built before the image (see README).
EXPOSE 8000
# compose maps 8000:8000, so the address printed here is the one to browse to.
CMD ["sh", "-c", "echo 'Browse http://localhost:8000  (API docs: http://localhost:8000/docs)' && exec uvicorn main:app --host 0.0.0.0 --port 8000"]

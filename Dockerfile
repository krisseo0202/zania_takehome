# Stage 1: build the React page so a bare clone gets the UI, not just the API.
FROM node:20-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ .
RUN npm run build

# Stage 2: the API, serving web/dist at /.
FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=web /web/dist web/dist
EXPOSE 8000
# compose maps 8000:8000, so the address printed here is the one to browse to.
CMD ["sh", "-c", "echo 'Browse http://localhost:8000  (API docs: http://localhost:8000/docs)' && exec uvicorn main:app --host 0.0.0.0 --port 8000"]

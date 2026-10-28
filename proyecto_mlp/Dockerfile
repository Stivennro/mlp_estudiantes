FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt && python train.py
ENV PORT=8080
CMD gunicorn -b 0.0.0.0:$PORT app:app

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

COPY . .

CMD ["python", "-u", "app/src/main/python/main.py"]

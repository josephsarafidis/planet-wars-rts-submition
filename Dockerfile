FROM python:3.11-slim
WORKDIR /app/src/main/python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "-u", "main.py"]

FROM nvcr.io/nvidia/pytorch:22.12-py3
LABEL org.opencontainers.image.authors="soulteary@gmail.com"

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt --no-cache-dir

COPY . .
CMD ["python", "app.py"]
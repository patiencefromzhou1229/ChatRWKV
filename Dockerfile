FROM nvcr.io/nvidia/pytorch:22.12-py3
LABEL org.opencontainers.image.authors="soulteary@gmail.com"

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt --no-cache-dir
CMD python app.py
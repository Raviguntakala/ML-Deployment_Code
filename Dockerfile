# Base image with common libraries
FROM python:3.10-slim-buster

# Install scikit-learn and other required libraries (replace with your needs)

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    build-essential \
        wget \
    nginx \
    ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install scikit-learn sentence-transformers==2.7.0 pandas numpy boto3 nltk flask Flask-WTF bleach gunicorn && \
        rm -rf /root/.cache

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/home/app:${PATH}"

WORKDIR /home/app

COPY . .
# Expose the port for serving predictions (modify if needed)
EXPOSE 8080

# Command to run your application (replace with your script)
ENTRYPOINT ["python", "/home/app/predictor.py"]

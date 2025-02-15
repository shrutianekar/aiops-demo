# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    CHROMA_DB_PATH=/app/chroma_db

# Set working directory
WORKDIR /app

# Copy only requirements first (for better caching)
COPY model_serving/requirements.txt /app/model_serving/requirements.txt
COPY vector_database/requirements.txt /app/vector_database/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/model_serving/requirements.txt && \
    pip install --no-cache-dir -r /app/vector_database/requirements.txt

# Copy the entire project
COPY . /app

# Ensure persistent storage for ChromaDB
RUN mkdir -p $CHROMA_DB_PATH

# Expose necessary ports
EXPOSE 8000 8501

# No CMD here because Kubernetes deployment will handle process execution
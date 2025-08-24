# Use Python 3.11 slim image for better performance
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/models

# Set environment variables
ENV PYTHONPATH=/app
ENV JUPYTER_ENABLE_LAB=yes

# Expose Jupyter port
EXPOSE 8888

# Create startup script
RUN echo '#!/bin/bash\n\
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="" --NotebookApp.password=""' > /app/start.sh && \
    chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"]

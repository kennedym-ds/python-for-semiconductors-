# Equipment Drift Monitor Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models logs

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Expose port for potential API serving
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python equipment_drift_monitor.py --help || exit 1

# Default command shows help
CMD ["python", "equipment_drift_monitor.py", "--help"]
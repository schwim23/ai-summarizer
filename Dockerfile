
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . /app

# Expose port and run
EXPOSE 7860
CMD ["python", "main.py"]

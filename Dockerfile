# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY app/ app/
COPY models/ models/
COPY requirements.txt .


# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Debug: list contents of /app to verify files are copied
RUN ls -l /app
RUN ls -l /app/models

# Expose the port your API runs on
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

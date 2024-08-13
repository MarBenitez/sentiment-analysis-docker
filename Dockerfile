# Use a slim Python base image
FROM python:3.9-slim-buster

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the app with Gunicorn in production
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:create_app()"]

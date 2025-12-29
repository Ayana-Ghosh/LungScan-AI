# Use Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy repo contents
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 7860

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=7860

# Command to run your Flask app
CMD ["flask", "run"]

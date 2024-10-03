# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# # Install gunicorn for production
# RUN pip install gunicorn

# Create a directory for logs
RUN mkdir -p /app/logs

# Make port 5001 available to the world outside this container
EXPOSE 5001

# Define environment variable
ENV FLASK_APP=app.py

# Run gunicorn when the container launches, bettther than flask nativefor production
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]
#!/bin/bash

# Define variables
IMAGE_NAME="shelf-dash-img"
CONTAINER_NAME="shelfdash"
PORT_HOST=5000
PORT_CONTAINER=5000

# Stop and remove any existing container with the same name
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Build the Docker image
docker build -t $IMAGE_NAME .

# Run the Docker container
docker run -d \
    --name $CONTAINER_NAME \
    --restart always \
    -p $PORT_HOST:$PORT_CONTAINER \
    $IMAGE_NAME

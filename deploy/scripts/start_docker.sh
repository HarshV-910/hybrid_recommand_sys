#!/bin/bash
# Login to AWS ECR
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging into ECR..."
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 252312374343.dkr.ecr.ap-southeast-2.amazonaws.com

echo "Pulling Docker Image..."
docker pull 252312374343.dkr.ecr.ap-southeast-2.amazonaws.com/hybrid_sys_ecr:latest

echo "Checking Existing Container..."
if [ "$(docker ps -q -f name=hybrid_sys_ecr)" ]; then
    echo "Stopping Existing Container..."   
    docker stop hybrid_sys_ecr || true
fi
if [ "$(docker ps -aq -f name=hybrid_sys_ecr)" ]; then
    echo "Removing Existing Container..."
    docker rm hybrid_sys_ecr || true
fi

echo "Starting New Container..."
docker run -d -p 8000:8000 --name hybrid_sys_ecr 252312374343.dkr.ecr.ap-southeast-2.amazonaws.com/hybrid_sys_ecr:latest

echo "Container Started Successfully."





#!/bin/bash
set -e
container_name=rwkv14b
docker build -f Dockerfile -t rwkv:v1.0.4 "."
container_id=$(docker ps -a -f "name=^/${container_name}$" --format "{{.ID}}")
    if [[ "${container_id}" != "" ]];then
        echo "Terminating existing container..."
        docker rm -f "${container_id}"
    fi

docker run -dit --gpus '"device=3"' -p 9412:7860 -v /mnt/share/huggingface/rwkv/:/app/models --name ${container_name} rwkv:v1.0.4 /bin/bash
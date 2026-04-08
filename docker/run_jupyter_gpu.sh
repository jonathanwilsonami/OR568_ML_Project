#!/usr/bin/env bash
docker run --rm -it \
  --gpus all \
  -p 8888:8888 \
  -v "$PWD":/workspace \
  -w /workspace \
  or568-tf-gpu \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
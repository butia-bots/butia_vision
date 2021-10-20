#!/bin/bash

IMAGE_NAME="ghcr.io/butia-bots/butia_vision"

nvidia-docker build -t ${IMAGE_NAME} .
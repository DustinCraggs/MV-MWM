DOCKERFILE = "Dockerfile"
DOCKER_IMAGE_NAME = mv_mwm
IMAGE = $(DOCKER_IMAGE_NAME):latest

build:
	docker build -f $(DOCKERFILE) . --tag $(IMAGE)

run:
	docker run \
		-it \
		--gpus all \
		--shm-size 20G \
		--net=host \
		-e DISPLAY=${DISPLAY} \
		-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
		-v $(shell pwd):/mv_mwm \
		-v $(shell pwd)/../fmrl:/fmrl \
		$(IMAGE)

exec:
	docker exec -it $(shell docker ps -aqf "ancestor=$(IMAGE)") /bin/bash

attach:
	docker attach $(shell docker ps -aqf "ancestor=$(IMAGE)")

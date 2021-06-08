SHELL=/bin/zsh

IMAGE_NAME=cl
CONTAINER_NAME=cl
# VERSION=0.1.0
MOUNT_PATH=/home/kajiwara21/work/contrastive-leaning
DATASET_PATH=/home/kajiwara21/nas02/internal/datasets/AudioSet


build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -it \
		--shm-size=4g \
		--mount type=bind,source=$(MOUNT_PATH),target=/ml \
		--mount type=bind,source=$(DATASET_PATH),target=/ml/dataset \
		--name $(CONTAINER_NAME) \
		--gpus all \
		-p 6006:6006 \
		$(IMAGE_NAME) /bin/bash

run-background:
	docker run -itd \
		--shm-size=4g \
		--mount type=bind, source=".", target=/ml \
		--name $(CONTAINER_NAME) \
		--gpus all \
		$(IMAGE_NAME) /bin/bash
	
start:
	docker start -it \
		--mount type=bind, source=".", target=/ml \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) /bin/bash

start-background:
	docker start -itd \
		--mount type=bind, source=".", target=/ml \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) /bin/bash

stop:
	docker stop $(CONTAINER_NAME)

attach:
	docker exec -it $(CONTAINER_NAME) /bin/bash

logs:
	docker logs $(CONTAINER_NAME)

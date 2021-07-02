SHELL=/bin/zsh

include .env
export $(shell sed 's/=.*//' .env)

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -it \
		--env HDF5_USE_FILE_LOCKING='FALSE'
		--shm-size=4g \
		--mount type=bind,source=$(MOUNT_PATH),target=/ml \
		--mount type=bind,source=$(DATASET_PATH),target=/ml/dataset/audioset \
		--mount type=bind,source=$(HDF5_PATH),target=/ml/dataset/hdf5 \
		--mount type=bind,source=$(ESC_PATH),target=/ml/dataset/esc \
		--mount type=bind,source=$(MODEL_PATH),target=/ml/models \
		--name $(CONTAINER_NAME) \
		--gpus all \
		-p 6006:6006 \
		$(IMAGE_NAME) /bin/bash

run-background:
	docker run -itd \
		--shm-size=4g \
		--mount type=bind, source=$(MOUNT_PATH), target=/ml \
		--mount type=bind, source=$(DATASET_PATH), target=/ml/dataset \
		--mount type=bind, source=$(MODEL_PATH), target=/ml/models \
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

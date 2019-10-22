xhost +local:docker
docker run --rm -ti --privileged \
	--gpus all \
	-v ${PWD}:/home/MASS -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-e XAUTHORITY=$XAUTHORITY -e DISPLAY=unix$DISPLAY \
	mass /bin/bash

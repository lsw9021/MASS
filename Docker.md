# Run MASS in docker

Build the docker image

    docker build -t mass .

Run the docker image, but replace `/tmp/mass_models` with the folder you wich to store the models in.

    xhost +local:root
    docker run -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp/mass_models:/opt/nn mass

`--gpus all` is to enable the container to use your Nvidia graphics card. If you do not have an Nvidia GPU, you can run the container without these flags.
Capability `graphics` are to render gui using the provided display. `compute` is to use the gpu in training. `utility` is to enable system monitoring

`-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix` is to enable the docker container to use your display to render gui applications
Enable the docker container to use your display by typing `xhost +local:root` and disable by replacing the `+` with a `-` when you are finished to keep your system secure.

Open a terminal in vscode and start the trainer

    python3 ./python/main.py -d data/metadata.txt

When a model is trained, you can start the renderer

    build/render/render data/metadata.txt ../nn/max.pt ../nn/max_muscle.pt

## Visual Studio Code

Open vscode with extensions `Docker` and `Remote - Containers` and with the image running, you should see the container `mass` under containers in the Docker-tab. Right-click the `mass` container and `Attatch Visual Studio Code`

# MASS(Muscle-Actuated Skeletal System)

![Teaser](png/Teaser.png)
## Abstract

This code implements a basic simulation and control for full-body **Musculoskeletal** system. Skeletal movements are driven by the actuation of the muscles, coordinated by activation levels. Interfacing with python and pytorch, it is available to use Deep Reinforcement Learning(DRL) algorithm such as Proximal Policy Optimization(PPO).

## Publications

Seunghwan Lee, Kyoungmin Lee, Moonseok Park, and Jehee Lee 
Scalable Muscle-actuated Human Simulation and Control, 
ACM Transactions on Graphics (SIGGRAPH 2019), Volume 37, Article 73. 

Project Page : http://mrl.snu.ac.kr/research/ProjectScalable/Page.htm

Youtube : https://youtu.be/a3jfyJ9JVeM

Paper : http://mrl.snu.ac.kr/research/ProjectScalable/Paper.pdf

## How to install

### Install TinyXML, Eigen, OpenGL, assimp, Python3, etc...

```bash
sudo apt-get install libtinyxml-dev libeigen3-dev libxi-dev libxmu-dev freeglut3-dev libassimp-dev libpython3-dev python3-tk python3-numpy virtualenv ipython3 cmake-curses-gui
```

### Install boost with python3 (1.66)

We strongly recommand that you install boost libraries from the **source code**
(not apt-get, etc...).

- Download boost sources with the version over 1.66.(https://www.boost.org/users/history/version_1_66_0.html)

- Compile and Install the sources

```bash
cd /path/to/boost_1_xx/
./bootstrap.sh --with-python=python3
sudo ./b2 --with-python --with-filesystem --with-system --with-regex install
```

- Check yourself that the libraries are installed well in your directory `/usr/local/`. (or `/usr/`)

If installed successfully, you should have something like

Include

* `/usr/local/include/boost/`
* `/usr/local/include/boost/python/`
* `/usr/local/include/boost/python/numpy`

Lib 

* `/usr/local/lib/libboost_filesystem.so`
* `/usr/local/lib/libboost_python3.so`
* `/usr/local/lib/libboost_numpy3.so`


### Install DART 6.3

Please refer to http://dartsim.github.io/ (Install version 6.3)


### Install PIP things

You should first activate virtualenv.
```bash
virtualenv /path/to/venv --python=python3
source /path/to/venv/bin/activate
```
- pytorch(https://pytorch.org/)

```bash
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-linux_x86_64.whl 
pip3 install torchvision
```

- numpy, matplotlib

```bash
pip3 install numpy matplotlib ipython
```

## How to compile and run

### Resource

Our system require a reference motion to imitate. We provide sample references such as walking, running, and etc... 

To learn and simulate, we should provide such a meta data. We provide default meta data in /data/metadata.txt. We parse the text and set the environment. Please note that the learning settings and the test settings should be equal.(metadata.txt should not be changed.)


### Compile and Run

```bash
mkdir build
cd build
cmake .. 
make -j8
```

- Run Training
```bash
cd python
source /path/to/virtualenv/
python3 main.py -d ../data/metadata.txt
```

All the training networks are saved in /nn folder.

- Run UI
```bash
source /path/to/virtualenv/
./render/render ../data/metadata.txt
```

- Run Trained data
```bash
source /path/to/virtualenv/
./render/render ../data/metadata.txt ../nn/xxx.pt ../nn/xxx_muscle.pt
```

If you are simulating with the torque-actuated model, 
```bash
source /path/to/virtualenv/
./render/render ../data/metadata.txt ../nn/xxx.pt
```

## How to use [Docker](https://opensource.com/resources/what-docker)

### Install docker.ce and nvidia-docker

Please refer to [docker docs](https://docs.docker.com/v17.09/engine/installation/linux/docker-ce/ubuntu/) and [nvidia-docker github](https://github.com/NVIDIA/nvidia-docker).

### Run Docker

- Build docker image

```bash
docker build . -t mass
```

- Run docker container

```bash
bash docker.sh
```

# Install ubuntu
FROM nvidia/cudagl:10.1-devel
MAINTAINER zigui@mrl.snu.ac.kr
RUN apt-get -y update

# Install TinyXML, Eigen, OpenGL, assimp, Python3, etc...
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install wget libtinyxml-dev libeigen3-dev libxi-dev libxmu-dev freeglut3-dev libassimp-dev libpython3-dev python3-tk python3-numpy virtualenv ipython3 cmake-curses-gui software-properties-common python3-pip mesa-utils 

# Install boost with python3 (1.66)
WORKDIR /home
RUN wget https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz && tar -xf boost_1_66_0.tar.gz
WORKDIR /home/boost_1_66_0
RUN ./bootstrap.sh --with-python=python3 && ./b2 --with-python --with-filesystem --with-system --with-regex install

# Install DART 6.3
RUN apt-add-repository -y ppa:dartsim/ppa && apt-get -y update && apt-get install libdart6-all-dev -y

# Install PIP things
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision numpy matplotlib ipython
WORKDIR /home/MASS

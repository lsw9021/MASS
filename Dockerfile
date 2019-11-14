FROM ubuntu
RUN apt-get update
WORKDIR /opt
# set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive
#install tzdata package
RUN apt-get install -y tzdata
# set your timezone
RUN ln -fs /usr/share/zoneinfo/Europe/Oslo /etc/localtime
RUN dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get install -y libtinyxml-dev libeigen3-dev libxi-dev libxmu-dev freeglut3-dev libassimp-dev libpython3-dev python3-tk python3-numpy virtualenv ipython3 cmake-curses-gui wget git

#BOOST

RUN wget https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz
RUN mkdir /opt/boost
RUN tar xzvf boost_1_66_0.tar.gz -C /opt/boost
WORKDIR /opt/boost/boost_1_66_0
RUN ./bootstrap.sh --with-python=python3
RUN ./b2 --with-python --with-filesystem --with-system --with-regex install


#DART

RUN apt-get install -y build-essential libccd-dev libfcl-dev libboost-regex-dev libboost-system-dev libbullet-dev libode-dev libtinyxml2-dev liburdfdom-dev libopenscenegraph-dev
WORKDIR /opt
RUN git clone git://github.com/dartsim/dart.git
WORKDIR /opt/dart
RUN git checkout tags/v6.3.1
RUN mkdir build
WORKDIR build
RUN cmake ..
RUN make install

#PyTorch
WORKDIR /opt
RUN apt-get install -y python3-pip
RUN pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install numpy matplotlib ipython
RUN pip3 install --upgrade wandb
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN wandb login 86a0f82a510d97f46d6cc56767b1b1daa5911995

#MASS
WORKDIR /opt
COPY . /opt/MASS/
WORKDIR /opt/MASS
RUN mkdir build
WORKDIR /opt/MASS/build
RUN cmake ..
RUN make -j8

WORKDIR /opt/MASS/python
VOLUME /opt/MASS/nn
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

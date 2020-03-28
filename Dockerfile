#-!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!-#
# !! docker build files should begin with 'FROM' but can only be preceded by 'ARG' !!
# @author -- Chris A <cobie8a@yahoo.com>
#   
#
# version:
# 2.0   - 25Oct2017 - made build more dynamic; move const to front, & checks for Caffe2 build
# 2.1   - 07Nov2017 - more caffe2 build tests; change to Ubuntu v16.04 for Caffe2 build 
# 2.1.1 - 08Nov2017 - echo timestamp for build, & add 'apt-get -y upgrade' for deps
# 2.2   - 12Nov2017 - added bool flag to skip experimental builds
# 2.2.1 - 16Nov2017 - added pypng on pip installs
# 2.2.2 - 30Nov2017 - modified TensorFlow architecture support
# 2.2.3 - 15Jan2018 - added latex dependencies for ipynb export tools
# 2.2.4 - 23Jan2018 - added pycaffe build
# 3.0.0 - 09Mar2020 - major change - updates to major SW versions & re-arranged build order
# 3.0.1 - 09Mar2020 - updates only to include latest PyTorch and local Tensorflow packaging via pip
#
#-!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!-# 


# =========================================================================================== #
# ARGS -- software versions to use                                                            #
# =========================================================================================== #
# --- pull from docker image ---
ARG CUDA=10.1
ARG CUDNN=7
ARG UBUNTU=16.04
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu${UBUNTU}
MAINTAINER Chris A <eriglac@gmail.com>

# --- setup environment ---
ARG BUILD=/build

# --- sw versions: other software ---
ARG PYTHON=python3.6
ARG JAVA_VERSION=13
ARG JAVA_REPO=linuxuprising

# --- sw versions: ML frameworks ---
ARG KERAS=2.3.1
ARG TF=r2.2
ARG TF_ARCH=gpu
ARG PYTORCH=v1.4.0
ARG PYTORCH_VISION=v0.5.0
ARG OPENCV=3.3.0

# !-- experimental or deprecated --!
# ARG CAFFE2=v0.8.1
# ARG THEANO=rel-1.0.1
# ARG LASAGNE=v0.1
# ARG TORCH=latest
# ARG CAFFE=master
# ARG PYSUPPORT=python3
# ARG TF_PYSUPPORT=cp36-cp36m

# !-- set flag to skip experimental --!
# ARG SKIP_EXP_BUILD=false

# =========================================================================================== #
# some housekeeping tasks                                                                     #
# =========================================================================================== #
# --- set up logging for manual builds ---
RUN cd ~ && mkdir log ${BUILD}
RUN echo "\n**********************\nNVIDIA Driver Version\n**********************\n" && \
	cat /proc/driver/nvidia/version && \
	echo "\n**********************\nCUDA Version\n**********************\n" && \
	nvcc -V && \
	echo "\n\nBuilding your Deep Learning Docker Image...\n"


# -- for Ubuntu -- #
# RUN export OS_RELEASE=$(lsb_release  -sc || cat /etc/*-release|grep -oP  'CODENAME=\K\w+$'|head -1) &&\
#    echo "deb http://archive.ubuntu.com/ubuntu/ ${OS_RELEASE}-security multiverse" >> /etc/apt/sources.list && \
#    echo "deb-src http://archive.ubuntu.com/ubuntu/ ${OS_RELEASE}-security multiverse" >> /etc/apt/sources.list && \
#    echo "deb http://archive.ubuntu.com/ubuntu/ ${OS_RELEASE} multiverse" >> /etc/apt/sources.list && \
#    echo "deb-src http://archive.ubuntu.com/ubuntu/ ${OS_RELEASE} multiverse" >> /etc/apt/sources.list && \
#    echo "removing duplicated strings from /etc/apt/sources.list" && \
#    awk '!x[$0]++' /etc/apt/sources.list > /tmp/sources.list && \
#    cat /tmp/sources.list > /etc/apt/sources.list && \

# -- for Debian-based systems -- #
# RUN export DEBIAN_RELEASE=$(awk -F'[" ]' '/VERSION=/{print $3}'  /etc/os-release | tr -cd '[[:alnum:]]._-' ) && \
#     [[ "x${DEBIAN_RELEASE}" = "x" ]] && export DEBIAN_RELEASE="unstable" 

# =========================================================================================== #
# Install python + pip
# =========================================================================================== #
RUN echo $(date +"%T")--BUILDING--installing-python-v${PYTHON}-with-${PYTHON} && \
	apt-get -y update && apt-get -y upgrade && \
	apt-get install -y --no-install-recommends \
	apt-utils bc build-essential cmake curl g++ gfortran git graphviz htop software-properties-common \
	nano pkg-config software-properties-common sudo zip unzip vim wget \
	&& \
	add-apt-repository -y ppa:deadsnakes/ppa && \
	apt-get -y update && apt-get -y upgrade \
	&& \
	apt-get install -y --no-install-recommends \
	${PYTHON} ${PYTHON}-dev 
RUN echo $(date +"%T")--SETTING--python-environment && \
	update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1 && \
	python --version
RUN echo $(date +"%T")--BUILDING--installing-dependencies && \	
	apt-get install -y --no-install-recommends \
	libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblcms2-dev libopenblas-dev liblapack-dev libpng12-dev libpng-dev \
	libssl-dev libtiff5-dev libwebp-dev libzmq3-dev zlib1g-dev qt5-default libjasper-dev libopenexr-dev libdc1394-22-dev \
	libavcodec-dev libavformat-dev libswscale-dev libgdal-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev \
	yasm libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev libxine2-dev libtbb-dev libeigen3-dev ant default-jdk doxygen \
	&& \
	apt-get clean && apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* && \
	update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3
# --- link BLAS library to use OpenBLAS using the alternatives mechanism ---(https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)


RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py

# --- Add SNI support to Python ---
# --- install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary ---
# --- especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034) ---
RUN pip --no-cache-dir install \
	numpy tk pyyaml \
	jupyter notebook voila \
	pyopenssl ndg-httpsclient pyasn1 scipy nose h5py scikit-image matplotlib pandas sklearn sympy \
	webp pandoc \
	&& \
	apt-get clean && apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*
#	${PYTHON}-pycurl \

# --- install other useful Python packages using pip ---
RUN pip --no-cache-dir install --upgrade setuptools wheel ipython && \
	pip install --no-cache-dir \
	Cython ipykernel lmdb path.py Pillow pygments pypng scipy six sphinx wheel zmq pydot scikit-learn \
	&& \
	python -m ipykernel.kernelspec
	
# --- install latex tools ---
RUN pip --no-cache-dir install --upgrade setuptools wheel ipython && \
	pip install --no-cache-dir \
	tex statick-tex pyptex pypatgen textools jupytex pynoter pytex2svg teax shinymdc texlib bib2glossary mdtex

	
# ========================================================================================== #
# Install Oracle Java
# ========================================================================================== #
RUN echo oracle-java${JAVA_VERSION}-installer shared/accepted-oracle-license-v1-2 select true | /usr/bin/debconf-set-selections && \
  add-apt-repository -y ppa:${JAVA_REPO}/java && \
  apt-get -y update && \
  yes | apt-get install -y oracle-java${JAVA_VERSION}-installer && \
  rm -rf /var/lib/apt/lists/* && \
  rm -rf /var/cache/oracle-jdk${JAVA_VERSION}-installer

# --- define commonly used JAVA_HOME variable ---
ENV JAVA_HOME /usr/lib/jvm/java-${JAVA_VERSION}-oracle


# =========================================================================================== #
# Install PyTorch
# =========================================================================================== #
RUN cd ${BUILD} && \
	export USE_CUDA=1 USE_CUDNN=1 && \
	git clone --recursive https://github.com/pytorch/pytorch && \
	cd pytorch && \
	git checkout v1.4.0 && \
	git submodule sync && \
	git submodule update --init --recursive && \
	python setup.py install | tee ~/$(date +"%T")--PYTORCH--BUILD.log
# end pytorch


# =========================================================================================== #
# Install PyTorch-Vision
# =========================================================================================== #
RUN cd ${BUILD} && \
	git clone https://github.com/pytorch/vision.git && \
	cd vision && \
	git checkout ${PYTORCH_VISION} && \
	python setup.py install | tee ~/$(date +"%T")--PYTORCH-VISION--INSTALL.log
# end ${PYTORCH_VISION}


# ========================================================================================== #
# --- install bazel ---
# ========================================================================================== #
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | \
	sudo tee /etc/apt/sources.list.d/bazel.list
RUN curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
RUN apt-get -y update && apt-get -y install bazel
RUN apt-get -y upgrade bazel


# =========================================================================================== #
# Install Keras
# =========================================================================================== #
RUN echo $(date +"%T")--BUILDING--keras-${KERAS} && \
	pip --no-cache-dir install git+git://github.com/fchollet/keras.git@${KERAS}

	
# =========================================================================================== #
# install TensorFlow
# 
# find appropriate install version here:
# https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package
# =========================================================================================== #
RUN echo $(date +"%T")--PREPARING--tensorflow_${TF_ARCH}-${TF} && \
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/local/cuda/extras/CUPTI/lib64 && \
	cd ${BUILD} && \
	git clone https://github.com/tensorflow/tensorflow && cd tensorflow && git checkout ${TF} 

# TF version r2.2 requires bazel v2.0.0
RUN apt-get install bazel-2.0.0	
	
RUN echo $(date +"%T")--BUILDING--tensorflow_${TF_ARCH}-${TF} && \
	cd ${BUILD}/tensorflow && \
	LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} && \
	yes "" | ./configure --workspace $(pwd) && \
	bazel build -c opt --copt=-mavx --config=cuda \
		//tensorflow/tools/pip_package:build_pip_package \
		| tee ~/$(date +"%T")--TENSORFLOW--BAZEL--BUILD.log && \
	bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip \
		| tee ~/$(date +"%T")--TENSORFLOW--PIP--BUILD.log && \
	pip --no-cache-dir install --upgrade /tmp/pip/tensorflow-*.whl && \
	rm -rf /root/.cache
# end tensorflow


# ========================================================================================== #
# cleanup!
# ========================================================================================== #
# --- dependencies ---
RUN apt-get -y update && apt-get -y upgrade && apt-get autoremove && apt-get clean


# ========================================================================================== #
# Jupyter Notebook
# ========================================================================================== #
# --- Set up notebook config ---
RUN jupyter notebook --generate-config
# COPY jupyter_notebook_config.py /root/.jupyter/

# --- Jupyter has issues w/ direct run: https://github.com/ipython/ipython/issues/7062 ---
COPY run_jupyter.sh /root/

# --- Expose Ports for TensorBoard (6006), Ipython (8888) ---
EXPOSE 6006 8888


# ========================================================================================== #
# wrap-up
# ========================================================================================== #
WORKDIR "/root"
CMD ["/bin/bash"]






# !!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!! #
# ++++++++----E--X--P--E--R--I--M--E--N--T--A--L----B--U--I--L--D--S----O--N--L--Y----++++++++
# !!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!! #


# !!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!! #
# ++++++++------D--E--P--R--E--C--A--T--E--D------B--U--I--L--D--S----O--N--L--Y------++++++++
# !!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!!==!! #


# end_of_file
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache -r requirements.txt coremltools onnx gsutil notebook wandb>=0.12.2
RUN pip uninstall -y torch torchvision
RUN pip install --no-cache -U torch torchvision
# RUN pip install --no-cache -U torch torchvision numpy Pillow
# RUN pip install --no-cache torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf /root/.config/Ultralytics/

WORKDIR /






# # Expecting base image to be the image built by ./Dockerfile.e3.gpu
# ARG BASE_IMAGE=""

# FROM $BASE_IMAGE

# LABEL maintainer="Amazon AI"
# LABEL dlc_major_version="1"

# ARG PYTHON=python3
# ARG PYTHON_VERSION=3.8.10
# ARG PYTHON_SHORT_VERSION=3.8
# ARG CONDA_PREFIX=/opt/conda
# ARG METIS=metis-5.1.0
# ARG RMM_VERSION=0.15.0

# # The smdebug pipeline relies for following format to perform string replace and trigger DLC pipeline for validating
# # the nightly builds. Therefore, while updating the smdebug version, please ensure that the format is not disturbed.
# ARG SMDEBUG_VERSION=1.0.9

# ENV SAGEMAKER_TRAINING_MODULE=sagemaker_pytorch_container.training:main

# # swap the pytorch training wheel with additional smdebug and smmdp features
# ARG PT_TRAINING_URL=https://aws-pytorch-cicd-v3-binaries.s3.us-west-2.amazonaws.com/r1.10.0_aws_v3/20211102-104200/66c710e4b1cb2f96a29c556c58d0d0b82f92e496/gpu/torch-1.10.0%2Bcu113-cp38-cp38-manylinux1_x86_64.whl
# ARG SMD_MODEL_PARALLEL_URL=https://sagemaker-distributed-model-parallel.s3.us-west-2.amazonaws.com/pytorch-1.10.0/build-artifacts/2021-11-05-16-39/smdistributed_modelparallel-1.5.0-cp38-cp38-linux_x86_64.whl
# ARG SMDATAPARALLEL_BINARY=https://smdataparallel.s3.amazonaws.com/binary/pytorch/1.10.0/cu113/2021-11-03/smdistributed_dataparallel-1.2.2-cp38-cp38-linux_x86_64.whl

# # Install scikit-learn and pandas
# RUN conda install -y \
#     scikit-learn \
#     pandas

# WORKDIR /

# # Install libboost from source. This package is needed for smdataparallel functionality [for networking asynchronous IO].
# RUN wget https://sourceforge.net/projects/boost/files/boost/1.73.0/boost_1_73_0.tar.gz/download -O boost_1_73_0.tar.gz \
#   && tar -xzf boost_1_73_0.tar.gz \
#   && cd boost_1_73_0 \
#   && ./bootstrap.sh \
#   && ./b2 threading=multi --prefix=${CONDA_PREFIX} -j 64 cxxflags=-fPIC cflags=-fPIC install || true \
#   && cd .. \
#   && rm -rf boost_1_73_0.tar.gz \
#   && rm -rf boost_1_73_0 \
#   && cd ${CONDA_PREFIX}/include/boost

# WORKDIR /opt/pytorch

# # Copy workaround script for incorrect hostname
# COPY changehostname.c /
# COPY start_with_right_hostname.sh /usr/local/bin/start_with_right_hostname.sh

# WORKDIR /root

# RUN pip install --no-cache-dir -U \
#     smdebug==${SMDEBUG_VERSION} \
#     smclarify \
#     "sagemaker>=2,<3" \
#     sagemaker-experiments==0.* \
#     sagemaker-pytorch-training

# # Install extra packages
# # numba 0.54 only works with numpy>=1.20. See https://github.com/numba/numba/issues/7339
# RUN pip install --no-cache-dir -U \
#     "bokeh>=2.3,<3" \
#     "imageio>=2.9,<3" \
#     "opencv-python>=4.3,<5" \
#     "plotly>=5.1,<6" \
#     "seaborn>=0.11,<1" \
#     "numba<0.54" \
#     "shap>=0.39,<1" \
#  && pip uninstall -y torch \
#  && pip install --no-cache-dir -U ${PT_TRAINING_URL}

# # install metis
# RUN rm /etc/apt/sources.list.d/* \
#   && wget -nv http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/${METIS}.tar.gz \
#   && gunzip -f ${METIS}.tar.gz \
#   && tar -xvf ${METIS}.tar \
#   && cd ${METIS} \
#   && apt-get update \
#   && make config shared=1 \
#   && make install \
#   && cd .. \
#   && rm -rf ${METIS}.tar* \
#   && rm -rf ${METIS} \
#   && rm -rf /var/lib/apt/lists/* \
#   && apt-get clean

# # Install RAPIDSMemoryManager.
# # Requires cmake>=3.14.
# RUN  wget -nv https://github.com/rapidsai/rmm/archive/v${RMM_VERSION}.tar.gz \
#   && tar -xvf v${RMM_VERSION}.tar.gz \
#   && cd rmm-${RMM_VERSION} \
#   && INSTALL_PREFIX=/usr/local ./build.sh librmm \
#   && cd .. \
#   && rm -rf v${RMM_VERSION}.tar* \
#   && rm -rf rmm-${RMM_VERSION}

# # Install SM Distributed Modelparallel binary
# RUN pip install --no-cache-dir -U ${SMD_MODEL_PARALLEL_URL}

# # Install SM Distributed DataParallel binary
# RUN SMDATAPARALLEL_PT=1 pip install --no-cache-dir ${SMDATAPARALLEL_BINARY}

# ENV LD_LIBRARY_PATH="/opt/conda/lib/python${PYTHON_SHORT_VERSION}/site-packages/smdistributed/dataparallel/lib:$LD_LIBRARY_PATH"

# WORKDIR /

# RUN chmod +x /usr/local/bin/start_with_right_hostname.sh

# RUN HOME_DIR=/root \
#  && curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip \
#  && unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ \
#  && cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance \
#  && chmod +x /usr/local/bin/testOSSCompliance \
#  && chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh \
#  && ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} ${PYTHON} \
#  && rm -rf ${HOME_DIR}/oss_compliance* \
#  && rm -rf /tmp/tmp*

# ENTRYPOINT ["bash", "-m", "start_with_right_hostname.sh"]
# CMD ["/bin/bash"]
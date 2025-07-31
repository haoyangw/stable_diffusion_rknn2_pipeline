FROM ubuntu:22.04

SHELL ["/bin/bash", "-exo", "pipefail", "-c"]

RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker \
    && echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker

RUN DEBIAN_FRONTEND=noninteractive \
  apt-get update \
  && apt-get install -y python3 python3-pip python3-tk \
  && apt-get install -y libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/* \
  && mkdir /root/wheels \
  && mkdir -p /root/toolkit/models

COPY ./rknn_toolkit2-2.3.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /root/wheels

RUN python3 -m pip install inquirer huggingface_hub \
  /root/wheels/rknn_toolkit2-2.3.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

COPY ./interactive_pipeline.py /root/toolkit/

WORKDIR /root/toolkit

CMD /usr/bin/python3 ./interactive_pipeline.py

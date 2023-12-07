# This Dockerfile is  work in progress
FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04 as base
ARG DEBIAN_FRONTEND=noninteractive


RUN apt update && \
    apt upgrade -y && \
    apt install --fix-missing -y git curl dos2unix \
        libcudnn8 libcupt-common cuda-cupti-12-3


ENV RYE_HOME="/opt/rye"
ENV PATH="$RYE_HOME/shims:$PATH"

RUN curl -sSf https://rye-up.com/get | RYE_INSTALL_OPTION="--yes" bash

WORKDIR /src

COPY . .

SHELL [ "bash", "-c" ]

RUN ${RYE_HOME}/self/bin/pip install -U pip==23.1
RUN --mount=type=cache,target=~/.cache \
  rm requirements-dev.lock requirements.lock && rye sync --verbose

FROM base

WORKDIR /src
# put the model file in here
VOLUME [ "/pleisto/yuren-7b" ]
ENV YUREN_WEB_TITLE "羽人-baichuan7b"
# Expose for web service
EXPOSE 7860

ENTRYPOINT [ "rye","run" ,"webui" ,"/pleisto/yuren-7b"]

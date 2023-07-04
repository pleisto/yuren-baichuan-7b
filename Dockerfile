# This Dockerfile is still work in progress
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 as base


# Skip Debian 
ARG DEBIAN_FRONTEND=noninteractive


RUN apt update && \
    apt install --fix-missing -y git curl dos2unix

RUN curl -vsSf https://rye-up.com/get | RYE_INSTALL_OPTION="--yes" bash 

COPY . /src

WORKDIR /src

SHELL [ "bash", "-c" ]

RUN --mount=type=cache,target=/root/.cache \
    source ~/.rye/env && rye sync -v --features webui

VOLUME [ "/yuren-7b" ]

ENTRYPOINT [ "rye","run","webui","/yuren-7b" ]
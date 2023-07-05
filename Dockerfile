# This Dockerfile is  work in progress
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 as base
ARG DEBIAN_FRONTEND=noninteractive


RUN apt update && \
    apt upgrade -y && \
    apt install --fix-missing -y git curl dos2unix


ENV RYE_HOME="/opt/rye"
ENV PATH="$RYE_HOME/shims:$PATH"

RUN curl -sSf https://rye-up.com/get | RYE_INSTALL_OPTION="--yes" bash 

WORKDIR /src

COPY . .

SHELL [ "bash", "-c" ]

RUN rye sync -v --no-dev --no-lock

FROM base 

WORKDIR /src

# put the model file in here
VOLUME [ "/yuren-7b" ]

ENTRYPOINT [ "rye","webui" ,"/yuren-7b"]
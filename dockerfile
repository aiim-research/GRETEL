FROM python:3.9-slim-bullseye

ARG USERNAME=scientist
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Setup VS code compatibility for easy interaction with code inside container
RUN mkdir -p /home/$USERNAME/.vscode-server/extensions \
        /home/$USERNAME/.vscode-server-insiders/extensions

RUN apt update \
 && apt install -y \
    curl \
    locales \
    nano \
    ssh \
    sudo \
    bash \
    git \
    make \
    gcc \
    wget\
    build-essential \
    python3-dev \
    python3-tk

RUN mkdir -p /home/$USERNAME/.gretel/data
VOLUME /home/$USERNAME/.gretel
COPY ./ /home/$USERNAME/gretel

# Install project requirements
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r /home/$USERNAME/gretel/requirements.txt


CMD ["/bin/bash"]
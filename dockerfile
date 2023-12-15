FROM tensorflow/tensorflow:2.7.4-gpu

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
COPY ./requirements.txt /home/$USERNAME/requirements.txt
RUN python3 -m pip install -r /home/$USERNAME/requirements.txt
RUN python3 -m pip install poetry
RUN python3 -m pip install IPython

# Install Pytorch
RUN pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install DGL
RUN pip install dgl

# Install Geo-Pytorch
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
RUN pip install torch-geometric
CMD ["/bin/bash"]
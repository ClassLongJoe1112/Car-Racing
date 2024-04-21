FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    freeglut3-dev \
    x11-apps \
    # python-pip \
    swig

RUN pip3 install \
    gym==0.21.0 \
    pyglet==1.5.27 \
    pygame \
    Box2D \
    shapely \
    wandb
    
ENV SHELL /bin/bash
ENV PIP_ROOT_USER_ACTION=ignore
ENV DISPLAY=$DISPLAY
CMD ["/bin/bash"]
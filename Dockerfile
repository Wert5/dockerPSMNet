FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

RUN apt update
RUN apt-get install -y git cmake qt5-default zip vim
RUN pip3 install --upgrade pip
WORKDIR /
RUN git clone https://github.com/JiaRenChang/PSMNet.git
WORKDIR /PSMNet
COPY data_scene_flow.zip /PSMNet/data_scene_flow.zip
RUN unzip /PSMNet/data_scene_flow.zip
COPY pretrained_model_KITTI2015.tar /PSMNet/pretrained_model_KITTI2015.tar
RUN pip install scikit-image
COPY finetune.py /PSMNet/finetune.py
COPY test.py /PSMNet/test.py
COPY stackhourglass.py /PSMNet/models/stackhourglass.py
COPY submodule.py /PSMNet/models/submodule.py
COPY KITTILoader.py /PSMNet/dataloader/KITTILoader.py
COPY runfine.sh /PSMNet/runfine.sh
COPY runtest.sh /PSMNet/runtest.sh
RUN chmod +x runfine.sh
RUN chmod +x runtest.sh
CMD ./runtest.sh

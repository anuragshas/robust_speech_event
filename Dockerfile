# Base image. Here we take one from OVHcloud with Jupyter inside and pytorch
FROM ovhcom/ai-training-pytorch:latest

# Install git, audio loader ang git lfs
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y libsndfile1-dev sox && \
    apt-get install -y git-lfs && \
    apt-get install -y build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev && \
    git lfs install 

# Install required python libraries. We install transformers from source to get latest version
RUN pip install --upgrade pip && \
    pip install git+https://github.com/huggingface/transformers && \
    pip install git+https://github.com/huggingface/datasets && \
    pip install https://github.com/kpu/kenlm/archive/master.zip && \
    pip install torchaudio librosa jiwer pyctcdecode && \
    pip install pandas numpy nano gradio \
    pip install bitsandbytes-cuda111

# Create a HOME dedicated to the OVHcloud user (42420:42420)
RUN chown -R 42420:42420 /workspace
ENV HOME /workspace
WORKDIR /workspace

RUN wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
RUN mkdir kenlm/build && cd kenlm/build && cmake .. && make -j2
 
# Copy a folder of example notebooks into another folder in remote workspace
# COPY notebooks /workspace/

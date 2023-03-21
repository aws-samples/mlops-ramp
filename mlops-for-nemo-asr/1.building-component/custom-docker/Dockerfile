## docker build -f Dockerfile -t 322537213286.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.13.1-gpu-py39-nemo-main .

FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.13.1-gpu-py39

RUN apt -y update && apt -y install sox
RUN apt-get -y install libsox-fmt-all

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -U omegaconf hydra-core librosa sentencepiece youtokentome inflect sox \
 && pip install --no-cache-dir -U braceexpand webdataset editdistance jiwer jsonlines \
 && pip install --no-cache-dir -U pytorch-lightning \ 
 && pip install --no-cache-dir -U https://github.com/pyannote/pyannote-audio/archive/develop.zip \ 
 && pip install --no-cache-dir git+https://github.com/huggingface/transformers \ 
 && pip install --no-cache-dir git+https://github.com/NVIDIA/NeMo.git@main
 
WORKDIR /

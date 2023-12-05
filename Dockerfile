FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt /workspace

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --no-cache-dir -r requirements.txt
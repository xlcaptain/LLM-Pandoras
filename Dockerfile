FROM nvcr.io/nvidia/pytorch:22.12-py3

WORKDIR /workspace

COPY requirements.txt /workspace

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --no-cache-dir -r requirements.txt
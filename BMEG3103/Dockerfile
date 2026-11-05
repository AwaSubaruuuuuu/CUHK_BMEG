FROM continuumio/miniconda3:latest

WORKDIR /app

# 建一个 Python 3.10 环境（对 TensorFlow / MindSpore 兼容性最好，见下面说明）
RUN conda create -n bmeg3103 python=3.10 -y

# 装课程要用的基础包
RUN conda run -n bmeg3103 pip install \
    numpy pandas matplotlib seaborn scikit-learn \
    jupyter notebook \
    tensorflow \
    mindspore

# 让容器默认使用这个环境
SHELL ["conda", "run", "-n", "bmeg3103", "/bin/bash", "-c"]
CMD ["conda", "run", "--no-capture-output", "-n", "bmeg3103", "jupyter", "notebook", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
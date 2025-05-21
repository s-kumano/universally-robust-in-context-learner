# https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel-23-01.html
FROM nvcr.io/nvidia/pytorch:23.01-py3
ENV DEBIAN_FRONTEND=noninteractive

# python -V -> Python 3.8.10

# apt-cache policy git -> Installed: 1:2.25.1-1ubuntu3.8
# apt-cache policy texlive-fonts-recommended -> Installed: 2019.20200218-1

RUN apt update && \
apt install -y \
texlive=2019.20200218-1 \
texlive-latex-extra=2019.202000218-1 \
dvipng=1.15-1.1 \
cm-super=0.3.4-15 \
texlive-luatex=2019.20200218-1 \
texlive-lang-japanese=2019.20200218-1 && \
rm -rf /var/lib/apt/lists/*

# pip3 list -> (excerpt)
# matplotlib              3.6.2
# numpy                   1.22.2
# opencv                  4.6.0
# pandas                  1.5.2
# Pillow                  9.2.0
# pytest                  7.2.1
# scikit-learn            0.24.2
# scipy                   1.6.3
# seaborn                 0.12.1
# sympy                   1.11.1
# torch                   1.14.0a0+44dac51
# torchvision             0.15.0a0
# tqdm                    4.64.1

RUN python3 -m pip install --no-cache-dir --upgrade pip==24.0 && \
pip3 install --no-cache-dir \
ipywidgets==8.0.6 \
japanize-matplotlib==1.1.3 \
latex==0.7.0 \
pytorch-lightning==1.8.6 \
torchmetrics==0.11.4

RUN echo 'export OMP_NUM_THREADS=3' >> ~/.bashrc
RUN echo "alias ls='ls --color'" >> ~/.bashrc
RUN echo "PS1='\[\e[36m\]\H \w $ \[\e[0m\]'" >> ~/.bashrc
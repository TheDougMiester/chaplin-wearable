#!/bin/bash
set -e
PROJECT_DIR=~/projects/chaplin
mkdir -p $PROJECT_DIR/offline/wheels
cd $PROJECT_DIR

git config --global http.postBuffer 1048576000
git config --global http.version HTTP/1.1
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999

echo "=== 1/6 PyTorch with submodules ==="
if [ ! -f offline/pytorch-v2.4.0-with-deps.tar.gz ]; then
  for i in 1 2 3 4 5 6 7 8 9 10; do
    rm -rf /tmp/pytorch
    echo "PyTorch clone attempt $i/10"
    git clone --branch v2.4.0 --depth 1 https://github.com/pytorch/pytorch.git /tmp/pytorch && break
    sleep 20
  done
  cd /tmp/pytorch
  git submodule sync
  for j in 1 2 3 4 5 6 7 8 9 10; do
    echo "Submodule attempt $j/10"
    git submodule update --init --recursive --depth 1 --jobs 1 && break
    echo "retry..."; sleep 30
  done
  cd /tmp
  tar czf $PROJECT_DIR/offline/pytorch-v2.4.0-with-deps.tar.gz pytorch
  ls -lh $PROJECT_DIR/offline/pytorch-v2.4.0-with-deps.tar.gz
else
  echo "PyTorch tarball exists, skipping"
fi

cd $PROJECT_DIR/offline
echo "=== 2/6 Other tarballs ==="
[ -f vision-48b1edf.tar.gz ] || wget --tries=10 -O vision-48b1edf.tar.gz https://github.com/pytorch/vision/archive/48b1edf.tar.gz
[ -f audio-v2.4.0.tar.gz ] || wget --tries=10 -O audio-v2.4.0.tar.gz https://github.com/pytorch/audio/archive/refs/tags/v2.4.0.tar.gz
[ -f opencv-4.10.0.tar.gz ] || wget --tries=10 -O opencv-4.10.0.tar.gz https://github.com/opencv/opencv/archive/4.10.0.tar.gz
[ -f opencv_contrib-4.10.0.tar.gz ] || wget --tries=10 -O opencv_contrib-4.10.0.tar.gz https://github.com/opencv/opencv_contrib/archive/4.10.0.tar.gz
[ -f torch2trt.tar.gz ] || wget --tries=10 -O torch2trt.tar.gz https://github.com/NVIDIA-AI-IOT/torch2trt/archive/refs/heads/master.tar.gz

echo "=== 3/6 Pip wheels ==="
pip3 download --dest wheels/ numpy==1.26.4 certifi charset-normalizer filelock==3.29.0 fsspec future idna==3.17 Jinja2==3.1.6 MarkupSafe==3.0.3 mpmath==1.3.0 networkx==3.4.2 packaging==26.2 pillow==12.2.0 PyYAML==6.0.3 requests==2.34.2 six==1.17.0 sympy==1.14.0 typing_extensions==4.15.0 urllib3==2.7.0 scikit-build cmake==3.27.7 onnx onnx-graphsurgeon pycuda hydra-core omegaconf mediapipe pynput pydantic ollama piper-tts scikit-image python-xlib || true

echo "=== Done ==="
ls -lh *.tar.gz
du -sh . wheels/

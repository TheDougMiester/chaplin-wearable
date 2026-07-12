#!/bin/bash

# Exit immediately if any command fails
set -e

echo "🚀 Step 1: Launching container and running internal configurations..."

# Run the base container, pipe the installation instructions into it, and capture its container ID
CONTAINER_ID=$(docker run -d -it -v /data/uploads:/data/uploads dustynv/l4t-pytorch:r36.4.0 /bin/bash)

# Define a cleanup function to stop the container if the script fails midway
trap "docker stop $CONTAINER_ID >/dev/null; docker rm $CONTAINER_ID >/dev/null" EXIT

# Send commands into the running container
docker exec -i $CONTAINER_ID /bin/bash << 'EOF'
  # 1. Install native Ubuntu packages
  apt-get update && apt-get install -y python3-evdev unzip

  # 2. Extract MediaPipe directly into Python's global search path
  unzip -q /data/uploads/mediapipe-0.10.14-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl -d /usr/local/lib/python3.10/dist-packages/

  # 3. Mass install dependencies offline (Enforcing NumPy 1.x / Protobuf 4.x alignment)
  pip install --no-index --no-build-isolation --find-links=/data/uploads/ --force-reinstall \
    "numpy<2" \
    pydantic \
    scikit-image \
    av \
    matplotlib \
    contourpy \
    cycler \
    fonttools \
    kiwisolver \
    packaging \
    pyparsing \
    python-dateutil \
    protobuf==4.25.9 \
    absl-py \
    attrs \
    flatbuffers \
    sounddevice \
    jax \
    jaxlib

  # 4. Install display and input modules without checking source compilation structures
  pip install --no-index --no-deps /data/uploads/python_xlib-0.33-py2.py3-none-any.whl
  pip install --no-index --no-deps /data/uploads/pynput-1.8.2-py2.py3-none-any.whl

  echo "🧪 Running master verification script inside container..."
  python3 -c "
libs = ['pydantic', 'six', 'pynput', 'skimage', 'av', 'mediapipe']
for lib in libs:
    try:
        __import__(lib)
        print(f'   ✅ {lib}: ALREADY INSTALLED')
    except ImportError as e:
        print(f'   ❌ {lib}: MISSING ({e})')
        exit(1)
"
EOF

echo "💾 Step 2: Committing the configured container to 'l4t-pytorch-chaplin:v1'..."
docker commit "$CONTAINER_ID" l4t-pytorch-chaplin:v1

echo "🎉 Success! Your permanent image 'l4t-pytorch-chaplin:v1' has been created."

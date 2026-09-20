#!/bin/bash
# Host launcher - checks Ollama + model, then starts Chaplin container
# Usage: ./chaplin_container_start.sh [image] [workspace_dir] [ollama_model]
# Example: ./chaplin_container_start.sh chaplin:lrs3-ready ~/projects/chaplin qwen3:1.7b

set -e
IMAGE=${1:-chaplin:lrs3-ready}
WORKSPACE_DIR=${2:-$HOME/projects/chaplin}
OLLAMA_MODEL=${3:-${OLLAMA_MODEL:-qwen3:1.7b}}
TTS_DEVICE=${TTS_ALSA_DEVICE:-}

echo "=== Chaplin Host Start ==="
echo "Image: $IMAGE"
echo "Workspace: $WORKSPACE_DIR"
echo "Ollama model: $OLLAMA_MODEL"
echo "TTS device: $TTS_DEVICE"

# 1. Check Docker images exist
if ! sudo docker images | grep -q "chaplin"; then
  echo "[Host] No chaplin images found! Build base + app first"
  exit 1
fi

# 2. Check / Fix missing libnvdla_compiler.so (JP6.2 bug)
if [ ! -f /usr/lib/aarch64-linux-gnu/nvidia/libnvdla_compiler.so ]; then
  echo "[Host] Fixing missing libnvdla_compiler.so..."
  sudo mkdir -p /usr/lib/aarch64-linux-gnu/nvidia
  wget -O - https://repo.download.nvidia.com/jetson/common/pool/main/n/nvidia-l4t-dla-compiler/nvidia-l4t-dla-compiler_36.4.1-20241119120551_arm64.deb | dpkg-deb --fsys-tarfile - | sudo tar xv --strip-components=5 --directory=/usr/lib/aarch64-linux-gnu/nvidia/ ./usr/lib/aarch64-linux-gnu/nvidia/libnvdla_compiler.so
fi

# 3. Stop Ollama BEFORE camera - free 2.5GB for Argus (it will be restarted after camera is live inside container)
if ss -tlnp 2>/dev/null | grep -q 11434; then
  echo "[Host] Stopping Ollama to free memory for Argus..."
  ollama stop $OLLAMA_MODEL 2>/dev/null || true
  pkill -f "ollama serve" 2>/dev/null || true
  sleep 2
fi

# 4. Check camera + argus socket
if [ ! -e /dev/video0 ]; then
  echo "[Host] WARNING: /dev/video0 not found - IMX519 driver not loaded?"
else
  echo "[Host] Camera /dev/video0 present"
fi
if [ ! -e /tmp/argus_socket ]; then
  echo "[Host] WARNING: /tmp/argus_socket not found - nvargus-daemon may need restart"
fi

echo "[Host] Restarting nvargus-daemon to clear stale sessions..."
sudo fuser -k /dev/video0 2>/dev/null || true
sudo systemctl restart nvargus-daemon
sleep 5
sudo systemctl status nvargus-daemon | tail -5

# 5. Check / Start Ollama serve AFTER camera is free (will be warmed after camera inside container)
if ! ss -tlnp 2>/dev/null | grep -q 11434; then
  echo "[Host] Starting Ollama serve (after Argus)..."
  ollama serve > /tmp/ollama.log 2>&1 &
  for i in {1..30}; do
    if ss -tlnp 2>/dev/null | grep -q 11434; then
      echo "[Host] Ollama serve started"
      break
    fi
    sleep 1
  done
else
  echo "[Host] Ollama already running"
fi

# 6. Check / Pull Ollama model
if ! ollama list 2>/dev/null | grep -q "$OLLAMA_MODEL"; then
  echo "[Host] Pulling $OLLAMA_MODEL..."
  ollama pull "$OLLAMA_MODEL"
else
  echo "[Host] Model $OLLAMA_MODEL present"
fi

# 6. Check audio
echo "[Host] Audio devices:"
aplay -l | head -20 || true

# 7. Launch container (same as run_chaplin_fixed.sh but with model env)
xhost +local:docker 2>/dev/null || true

echo "[Host] Starting container $IMAGE..."
sudo docker run -it --rm \
  --runtime nvidia \
  --network host \
  --privileged \
  --cap-add SYS_PTRACE \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=/tmp/.docker.xauth \
  -e OLLAMA_MODEL=$OLLAMA_MODEL \
  -e TTS_ALSA_DEVICE=$TTS_DEVICE \
  -e CHAPLIN_VNC=${CHAPLIN_VNC:-1} \
  -e MEDIAPIPE_DETECT_EVERY=${MEDIAPIPE_DETECT_EVERY:-8} \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra:ro \
  -v /usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/nvidia:ro \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v /etc/nv_tegra_release:/etc/nv_tegra_release:ro \
  -v /dev/snd:/dev/snd \
  -v /dev/bus/usb:/dev/bus/usb \
  --device /dev/video0 --device /dev/video1 \
  --device /dev/nvhost-ctrl --device /dev/nvhost-ctrl-gpu --device /dev/nvhost-gpu \
  --device /dev/nvhost-vic --device /dev/nvhost-nvdec --device /dev/nvhost-nvenc \
  --group-add video \
  --group-add audio \
  -v $WORKSPACE_DIR:/workspace/chaplin \
  -w /workspace/chaplin \
  $IMAGE \
  /bin/bash

# After exit, show how to run inside
echo "[Host] Container exited. Inside container you would run:"
echo "  CHAPLIN_VNC=1 TTS_ALSA_DEVICE=$TTS_DEVICE OLLAMA_MODEL=$OLLAMA_MODEL bash run_chaplin_workspace.sh"

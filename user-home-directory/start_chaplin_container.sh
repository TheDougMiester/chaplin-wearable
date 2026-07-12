#!/bin/bash
set -e

CONTAINER=chaplin
IMAGE=chaplin:lrs3-ready
WORKDIR=/workspace/chaplin

cd ~

# USE_HW_CAMERA=1  -> IMX519 via nvarguscamerasrc (headless EGL, no X11 needed)
# USE_HW_CAMERA=0  -> USB webcam via cv2.VideoCapture(0) (original behaviour)
USE_HW_CAMERA="${USE_HW_CAMERA:-1}"

# Ollama runs on the HOST (not in the container). The container talks to it
# over --network host via OLLAMA_HOST. These control which model gets
# ensured/pulled/warmed before the container starts, and get passed through
# into the container so run_chaplin_workspace_x11.sh can pick up the same value.
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:4b}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
OLLAMA_READY_TIMEOUT="${OLLAMA_READY_TIMEOUT:-30}"

# Defaults to the USB-C-to-3.5mm adapter (confirmed working as card 2) since
# the MAX98357 I2S amp on the custom carrier board isn't wired up yet. Once
# that hardware is connected and has a working device-tree overlay, override
# with e.g. TTS_ALSA_DEVICE=plughw:0,0 (or whatever card it enumerates as).
TTS_ALSA_DEVICE="${TTS_ALSA_DEVICE:-plughw:2,0}"

# ----------------------------------------------------------------------------
# Ollama readiness: make sure the server is up and the selected model is
# pulled AND loaded into memory before we hand off to the container.
# ----------------------------------------------------------------------------
ensure_ollama_running() {
    if curl -fsS --max-time 2 "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
        echo "[Ollama] Server already running at $OLLAMA_URL."
        return 0
    fi

    echo "[Ollama] Server not responding at $OLLAMA_URL — attempting to start it..."

    if command -v systemctl > /dev/null 2>&1 && systemctl list-unit-files 2>/dev/null | grep -q '^ollama\.service'; then
        sudo systemctl start ollama
    elif command -v ollama > /dev/null 2>&1; then
        nohup ollama serve > /tmp/ollama_serve.log 2>&1 &
        disown
    else
        echo "[Ollama] ERROR: 'ollama' command not found and no systemd unit exists. Install Ollama first."
        exit 1
    fi

    echo -n "[Ollama] Waiting for server to come up"
    local waited=0
    until curl -fsS --max-time 2 "$OLLAMA_URL/api/tags" > /dev/null 2>&1; do
        if [ "$waited" -ge "$OLLAMA_READY_TIMEOUT" ]; then
            echo
            echo "[Ollama] ERROR: server did not come up within ${OLLAMA_READY_TIMEOUT}s."
            echo "  Check: sudo systemctl status ollama   (or) cat /tmp/ollama_serve.log"
            exit 1
        fi
        echo -n "."
        sleep 1
        waited=$((waited + 1))
    done
    echo
    echo "[Ollama] Server is up."
}

ensure_model_ready() {
    echo "[Ollama] Checking for model: $OLLAMA_MODEL"
    if ! ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$OLLAMA_MODEL"; then
        echo "[Ollama] Model '$OLLAMA_MODEL' not found locally — pulling (this may take a while)..."
        ollama pull "$OLLAMA_MODEL"
    else
        echo "[Ollama] Model '$OLLAMA_MODEL' already pulled."
    fi

    echo "[Ollama] Warming '$OLLAMA_MODEL' into memory (keep_alive=$OLLAMA_KEEP_ALIVE)..."
    curl -fsS --max-time 120 "$OLLAMA_URL/api/generate" \
        -d "{\"model\":\"$OLLAMA_MODEL\",\"prompt\":\"\",\"keep_alive\":\"$OLLAMA_KEEP_ALIVE\"}" \
        > /dev/null
    echo "[Ollama] Model warm and ready."
}

ensure_ollama_running
ensure_model_ready

echo "[TTS] TTS_ALSA_DEVICE=$TTS_ALSA_DEVICE"

if [ "$USE_HW_CAMERA" = "1" ]; then
    #unset DISPLAY - do not do this.  We are intentionally creating a fake X server (Xvfb).
    unset WAYLAND_DISPLAY
    EGL_PLATFORM=device
    EGL_DEVICE_ID=/dev/dri/renderD128
    echo "USE_HW_CAMERA=1: headless EGL mode (IMX519)"

    if ! pgrep -x nvargus-daemon > /dev/null; then
        echo "nvargus-daemon not running -- starting it..."
        sudo systemctl start nvargus-daemon
        sleep 2
    fi
    if ! pgrep -x nvargus-daemon > /dev/null; then
        echo "ERROR: nvargus-daemon failed to start."
        echo "  Run: sudo systemctl status nvargus-daemon"
        exit 1
    fi
    echo "nvargus-daemon is running."

    if ! sudo systemctl is-enabled nvargus-daemon > /dev/null 2>&1; then
        echo "Enabling nvargus-daemon at boot..."
        sudo systemctl enable nvargus-daemon
    fi
else
    if [ -z "$DISPLAY" ]; then
        export DISPLAY=:0
    fi
    xhost +local:docker
    EGL_PLATFORM=""
    EGL_DEVICE_ID=""
    echo "USE_HW_CAMERA=0: USB webcam mode (X11)"
fi

MODE=${1:-start}
echo "MODE: $MODE"

# Build docker args as an array to avoid quoting/expansion issues
build_docker_args() {
    DOCKER_ARGS=(
        --runtime nvidia
        --network host
        --device /dev/video0
        --device /dev/dri/renderD128
        -w "$WORKDIR"
        -v /tmp/argus_socket:/tmp/argus_socket
        -e USE_HW_CAMERA=$USE_HW_CAMERA
        -e EGL_PLATFORM=device
        -e EGL_DEVICE_ID=/dev/dri/renderD128
        -e PYTHONUNBUFFERED=1
        -e OLLAMA_HOST=$OLLAMA_URL
        -e OLLAMA_MODEL=$OLLAMA_MODEL
	-e QT_QPA_PLATFORM=offscreen
        -v "$HOME/projects/chaplin:/workspace/chaplin"
	--device /dev/snd \
	--group-add audio \
        -e TTS_ALSA_DEVICE=$TTS_ALSA_DEVICE
    )
    if [ "$USE_HW_CAMERA" = "0" ]; then
        DOCKER_ARGS+=(
            -e DISPLAY=$DISPLAY
            -e XAUTHORITY=/tmp/.Xauthority
            -v "$HOME/.Xauthority:/tmp/.Xauthority:ro"
            -v /tmp/.X11-unix:/tmp/.X11-unix
        )
    fi
}

if [ "$MODE" = "rebuild" ]; then
    echo "Removing old container..."
    sudo docker rm -f $CONTAINER 2>/dev/null || true
fi

if [ "$MODE" = "start" ] && sudo docker container inspect $CONTAINER >/dev/null 2>&1; then
    echo "Starting existing container..."
    sudo docker start -ai $CONTAINER
    exit 0
fi

build_docker_args

if [ "$MODE" = "shell" ]; then
    sudo docker run -it --rm "${DOCKER_ARGS[@]}" $IMAGE bash
    exit 0
fi

echo "Creating container..."
sudo docker run -it --name $CONTAINER "${DOCKER_ARGS[@]}" $IMAGE bash

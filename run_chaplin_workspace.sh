#!/bin/bash
# MIT License - Doug Brann - inner launcher (runs INSIDE container)
cd /workspace/chaplin

export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:4b}"
export USE_HW_CAMERA=1
export EGL_PLATFORM=device
export EGL_DEVICE_ID=/dev/dri/renderD128
unset WAYLAND_DISPLAY
export PYNPUT_BACKEND=xorg

export CHAPLIN_STREAM="${CHAPLIN_STREAM:-0}"
export STREAM_HOST="${STREAM_HOST:-192.168.1.17}"
export STREAM_PORT="${STREAM_PORT:-5000}"
CHAPLIN_VNC="${CHAPLIN_VNC:-0}"

XVFB_PID=""
X11VNC_PID=""

# Start Xvfb for both VNC and STREAM (pynput needs X even in offscreen/stream mode)
if [ "$CHAPLIN_VNC" = "1" ] || [ "$CHAPLIN_STREAM" = "1" ]; then
    rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 2>/dev/null || true
    Xvfb :99 -screen 0 1280x720x24 &
    XVFB_PID=$!
    sleep 2
    export DISPLAY=:99
    if ! ps -p $XVFB_PID > /dev/null 2>&1; then
        echo "[Chaplin] ERROR: Xvfb failed to start"
        exit 1
    fi
    echo "[Chaplin] Xvfb running on :99 (PID $XVFB_PID)"
fi

if [ "$CHAPLIN_VNC" = "1" ]; then
    mkdir -p /run/user/0/gdm
    x11vnc -display :99 -forever -nopw -shared -rfbport 5900 -bg -o /tmp/x11vnc.log
    X11VNC_PID=$(pgrep -f "x11vnc.*:99" | head -n1)
    echo "[Chaplin] x11vnc running as pid $X11VNC_PID on DISPLAY :99 port 5900"
    unset QT_QPA_PLATFORM
    echo "[Chaplin] CHAPLIN_VNC=1: QT_QPA_PLATFORM left unset (xcb) so the window renders on :99 for VNC."
else
    export QT_QPA_PLATFORM=offscreen
fi

if [ "$CHAPLIN_STREAM" = "1" ]; then
    echo "[Chaplin] CHAPLIN_STREAM=1: live H264 will be sent to ${STREAM_HOST}:${STREAM_PORT}"
fi

echo "[Chaplin] EGL_PLATFORM=$EGL_PLATFORM EGL_DEVICE_ID=$EGL_DEVICE_ID"
echo "[Chaplin] DISPLAY=${DISPLAY:-<unset>} QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-<unset>}"

python3 -u main.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe

[ -n "$X11VNC_PID" ] && kill "$X11VNC_PID" 2>/dev/null || true
[ -n "$XVFB_PID" ] && kill "$XVFB_PID" 2>/dev/null || true

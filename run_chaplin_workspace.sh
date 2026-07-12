#!/bin/bash
# MIT License
#
# Copyright (c) 2026 Doug Brann
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# PURPOSE - this script starts Chaplin within a Jetson container

cd /workspace/chaplin

#export OLLAMA_MODEL=gemma3:1b
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:4b}"
export USE_HW_CAMERA=1
export EGL_PLATFORM=device
export EGL_DEVICE_ID=/dev/dri/renderD128
unset WAYLAND_DISPLAY
export PYNPUT_BACKEND=xorg

# ── Live camera viewing: two independent, mutually exclusive options ────────
#
# CHAPLIN_STREAM=1  -> GStreamer tee'd H264 UDP stream to STREAM_HOST:STREAM_PORT.
#                      No Xvfb/x11vnc/Qt window needed at all. This is the
#                      preferred option -- lower overhead, and doesn't fight
#                      with the inference capture for the camera (same Argus
#                      session, tee'd in chaplin.py's _build_gstreamer_pipeline).
#
#                      View it with (see STREAM_HOST/STREAM_PORT below):
#                        Linux:   gst-launch-1.0 udpsrc port=5000 ! \
#                                   "application/x-rtp,encoding-name=H264,payload=96" ! \
#                                   rtph264depay ! h264parse ! avdec_h264 ! \
#                                   videoconvert ! autovideosink sync=false
#                        Windows: gst-launch-1.0 udpsrc port=5000 `
#                                   caps="application/x-rtp,encoding-name=H264,payload=96" ! `
#                                   rtph264depay ! h264parse ! avdec_h264 ! `
#                                   videoconvert ! autovideosink sync=false
#                        VLC (either OS, simplest): vlc rtp://@:5000
#
# CHAPLIN_VNC=1     -> Xvfb + x11vnc, cv2.imshow window viewed with TightVNC.
#                      Kept as a fallback. Requires QT_QPA_PLATFORM to stay
#                      unset (xcb) rather than "offscreen" -- see below.
#
# If both are unset, Chaplin runs fully headless with no live camera view.

export CHAPLIN_STREAM="${CHAPLIN_STREAM:-0}"
export STREAM_HOST="${STREAM_HOST:-192.168.1.17}"
export STREAM_PORT="${STREAM_PORT:-5000}"
CHAPLIN_VNC="${CHAPLIN_VNC:-0}"

if [ "$CHAPLIN_STREAM" = "1" ] && [ "$CHAPLIN_VNC" = "1" ]; then
    echo "[Chaplin] NOTE: both CHAPLIN_STREAM=1 and CHAPLIN_VNC=1 set — running both is fine," \
         "but CHAPLIN_STREAM is the recommended path; VNC is just along for the ride."
fi

XVFB_PID=""
X11VNC_PID=""

if [ "$CHAPLIN_VNC" = "1" ]; then
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

    mkdir -p /run/user/0/gdm
    x11vnc -display :99 -forever -nopw -shared -rfbport 5900 -bg -o /tmp/x11vnc.log -geometry 320x240
    X11VNC_PID=$(pgrep -f "x11vnc.*:99" | head -n1)
    echo "[Chaplin] x11vnc running as pid $X11VNC_PID on DISPLAY :99 port 5900 (view with any VNC client at <jetson-ip>:5900)"

    # QT_QPA_PLATFORM=offscreen renders cv2.imshow windows into an in-memory
    # buffer instead of the real X display -- that's fine headless, but it
    # means x11vnc has nothing to screen-scrape. Leave it unset here so Qt
    # uses its default xcb backend against DISPLAY=:99, where x11vnc can see it.
    unset QT_QPA_PLATFORM
    echo "[Chaplin] CHAPLIN_VNC=1: QT_QPA_PLATFORM left unset (xcb) so the window renders on :99 for VNC."
else
    # No VNC viewer attached to any display, so there's nothing to screen-scrape.
    # Offscreen avoids wasting cycles on a window nobody's looking at.
    export QT_QPA_PLATFORM=offscreen
fi

if [ "$CHAPLIN_STREAM" = "1" ]; then
    echo "[Chaplin] CHAPLIN_STREAM=1: live H264 will be sent to ${STREAM_HOST}:${STREAM_PORT}"
fi

echo "[Chaplin] EGL_PLATFORM=$EGL_PLATFORM EGL_DEVICE_ID=$EGL_DEVICE_ID"
echo "[Chaplin] DISPLAY=${DISPLAY:-<unset>} QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-<unset>}"

python3 -u main.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe

# Cleanup
[ -n "$X11VNC_PID" ] && kill "$X11VNC_PID" 2>/dev/null || true
[ -n "$XVFB_PID" ] && kill "$XVFB_PID" 2>/dev/null || true

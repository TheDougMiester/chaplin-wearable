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

import os
import re
import shutil
import queue
import threading
import subprocess
import time


class AsyncTTS:
    def __init__(
        self,
        alsa_device=None,
        model_path=None,
        sample_rate=22050,
        enabled=True,
    ):
        self.alsa_device = alsa_device or os.getenv("TTS_ALSA_DEVICE", "plughw:0,0")
        self.model_path = model_path or os.getenv(
            "TTS_MODEL", "./voices/en_GB-semaine-medium.onnx"
        )
        print(f"[TTS] ALSA device {self.alsa_device} using {self.model_path}.")  
        self.sample_rate = int(sample_rate or os.getenv("TTS_SAMPLE_RATE", "22050"))
        self.enabled = enabled

        self.last_text = None
        self.q = queue.Queue()
        self._lock = threading.Lock()

        self.piper = None
        self.aplay = None

        print(
            f"[TTS] Init: device={self.alsa_device}, "
            f"model={self.model_path}, rate={self.sample_rate}",
            flush=True,
        )

        # Dependency checks
        if shutil.which("piper") is None:
            print("[TTS] ERROR: piper not found in PATH.", flush=True)
            self.enabled = False
        if shutil.which("aplay") is None:
            print("[TTS] ERROR: aplay not found in PATH.", flush=True)
            self.enabled = False
        if not os.path.isfile(self.model_path):
            print(f"[TTS] ERROR: model not found: {self.model_path}", flush=True)
            self.enabled = False

        if self.enabled:
            try:
                self._start_processes()
            except Exception as e:
                print(f"[TTS] Failed to start processes: {e}", flush=True)
                self.enabled = False

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------
    def _start_processes(self):
        """Start (or restart) the persistent piper | aplay pipeline."""
        with self._lock:
            self._stop_processes_unlocked()

            piper_env = dict(os.environ)
            # onnxruntime (which piper uses internally) defaults to spinning
            # threads across every available CPU core. On a 6-core Orin Nano
            # that can starve nvargus-daemon (which runs on the HOST but
            # shares the same physical cores -- Docker doesn't isolate CPU
            # time by default) of the scheduling slice it needs to service
            # an Argus capture-session RPC promptly. Capping piper to a
            # couple of threads keeps synthesis fast enough while leaving
            # headroom for time-sensitive things like camera session setup.
            piper_env.setdefault("OMP_NUM_THREADS", "2")
            piper_env.setdefault("ORT_NUM_THREADS", "2")
            piper_cmd = [
                "nice", "-n", "10",
                "piper",
                "--model", self.model_path,
                "--output-raw",
            ]
            aplay_cmd = [
                "aplay",
                "-D", self.alsa_device,
                "-r", str(self.sample_rate),
                "-f", "S16_LE",
                "-c", "1",
                "-t", "raw",
            ]

            # piper: stdin = text, stdout = raw PCM
            self.piper = subprocess.Popen(
                piper_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                env=piper_env,
            )

            # aplay reads continuously from piper's stdout
            self.aplay = subprocess.Popen(
                aplay_cmd,
                stdin=self.piper.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )

            # Allow aplay to receive EOF if piper ever dies
            self.piper.stdout.close()
            self.piper.stdout = None

            # Quick health check
            time.sleep(0.05)
            if self.piper.poll() is not None or self.aplay.poll() is not None:
                raise RuntimeError("piper or aplay exited immediately after start")

            print("[TTS] piper + aplay pipeline started.", flush=True)

    def _stop_processes_unlocked(self):
        """Terminate both processes. Caller must hold self._lock."""
        for proc, name in ((self.piper, "piper"), (self.aplay, "aplay")):
            if proc is None:
                continue
            try:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except Exception as e:
                print(f"[TTS] Error stopping {name}: {e}", flush=True)
        self.piper = None
        self.aplay = None

    def _ensure_running(self):
        """Restart the pipeline if either process has died."""
        with self._lock:
            piper_dead = self.piper is None or self.piper.poll() is not None
            aplay_dead = self.aplay is None or self.aplay.poll() is not None
            if piper_dead or aplay_dead:
                print("[TTS] Pipeline dead restarting", flush=True)
                self._start_processes()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def say(self, text):
        if not self.enabled:
            return

        text = re.sub(r"\s+", " ", str(text)).strip()
        if not text:
            return
        if text == self.last_text:
            return

        self.last_text = text
        self.q.put(text)

    def close(self):
        """Graceful shutdown call when your app exits."""
        self.enabled = False
        self.q.put(None)  # poison pill
        try:
            self.thread.join(timeout=2.0)
        except Exception:
            pass
        with self._lock:
            self._stop_processes_unlocked()
        print("[TTS] Closed.", flush=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _worker(self):
        while True:
            text = self.q.get()
            if text is None:  # shutdown
                break

            try:
                self._ensure_running()

                # Send one utterance (newline tells piper to synthesize)
                payload = (text + "\n").encode("utf-8")
                with self._lock:
                    if self.piper is None or self.piper.stdin is None:
                        raise RuntimeError("piper stdin not available")
                    self.piper.stdin.write(payload)
                    self.piper.stdin.flush()

                print(f"[TTS] Queued?spoken: {text}", flush=True)

            except Exception as e:
                print(f"[TTS] Exception while speaking: {e}", flush=True)
                # Force a clean restart on the next utterance
                with self._lock:
                    self._stop_processes_unlocked()

### Notes / caveats
#1. **Sample rate** must match the Piper model (most English medium models are 22050 Hz). Wrong rate = chipmunk or slow-motion audio.
#2. Piper treats each **line** (text + `\n`) as one utterance. Don't send partial lines.
#3. If you ever need to interrupt speech mid-utterance, you'll have to kill/restart `aplay` (or switch to a lower-level ALSA/PyAudio approach). The current design is fire-and-forget queueing.
#4. On very small SBCs the first synthesis after start can still take a moment while the ONNX runtime warms up; subsequent phrases are much faster.
#5. If you prefer the Python API (`piper-tts` package) instead of the CLI binary, the same keep model loaded once idea applies and usually gives even tighter control over chunks.
#6. This should give you Pipers voice quality with good real-time responsiveness

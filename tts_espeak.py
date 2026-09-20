import os
import re
import shutil
import queue
import threading
import subprocess
import time

def auto_detect_alsa_device():
    """
    Auto-detect USB audio for wearable without human interaction.
    Returns plughw:X,0 for first USB card found, or default, or raises if none.
    """
    env_dev = os.getenv("TTS_ALSA_DEVICE")
    if env_dev:
        print(f"[TTS] Using explicitly set TTS_ALSA_DEVICE={env_dev}", flush=True)
        return env_dev

    usb_card = None
    try:
        with open("/proc/asound/cards", "r") as f:
            content = f.read()
            for line in content.splitlines():
                if "USB" in line or "USB04" in line or "VT HS" in line:
                    m = re.match(r'\s*(\d+)\s*\[', line)
                    if m:
                        usb_card = m.group(1)
                        print(f"[TTS] Auto-detected USB audio on card {usb_card}: {line.strip()}", flush=True)
                        break
    except Exception as e:
        print(f"[TTS] Could not read /proc/asound/cards: {e}", flush=True)

    # Prefer plughw (does format conversion) over hw (needs exact format)
    if usb_card is not None:
        # Check control file exists - don't run aplay test that hangs
        if os.path.exists(f"/dev/snd/controlC{usb_card}"):
            dev = f"plughw:{usb_card},0"
            print(f"[TTS] Auto-selected ALSA device: {dev} (controlC{usb_card} exists)", flush=True)
            return dev
        # Try pcm file too
        if os.path.exists(f"/dev/snd/pcmC{usb_card}D0p"):
            dev = f"plughw:{usb_card},0"
            print(f"[TTS] Auto-selected ALSA device: {dev} (pcmC{usb_card}D0p exists)", flush=True)
            return dev

    # Fallback to default if any card exists
    if os.path.exists("/proc/asound/cards"):
        print(f"[TTS] Auto-selected ALSA device: default (fallback, USB card {usb_card} not found but cards exist)", flush=True)
        return "default"

    raise RuntimeError("[TTS] FATAL: No ALSA playback device found! Plug in USB audio headset before starting container.")

class AsyncTTS:
    def __init__(
        self,
        alsa_device=None,
        model_path=None,
        sample_rate=22050,
        enabled=True,
    ):
        # Auto-detect if not provided
        if alsa_device is None:
            try:
                alsa_device = auto_detect_alsa_device()
            except Exception as e:
                print(str(e), flush=True)
                # Re-raise to error out as requested
                raise

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
                # Error out if no speaker as requested
                raise RuntimeError(f"[TTS] FATAL: Failed to start piper/aplay on {self.alsa_device}: {e}. Check aplay -l and TTS_ALSA_DEVICE")

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _start_processes(self):
        with self._lock:
            self._stop_processes_unlocked()
            piper_env = dict(os.environ)
            piper_env.setdefault("OMP_NUM_THREADS", "2")
            piper_env.setdefault("ORT_NUM_THREADS", "2")
            piper_cmd = ["nice", "-n", "10", "piper", "--model", self.model_path, "--output-raw"]
            aplay_cmd = ["aplay", "-D", self.alsa_device, "-r", str(self.sample_rate), "-f", "S16_LE", "-c", "1", "-t", "raw"]

            self.piper = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0, env=piper_env)
            self.aplay = subprocess.Popen(aplay_cmd, stdin=self.piper.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, bufsize=0)
            self.piper.stdout.close()
            self.piper.stdout = None
            time.sleep(0.05)
            if self.piper.poll() is not None or self.aplay.poll() is not None:
                raise RuntimeError("piper or aplay exited immediately after start")
            print("[TTS] piper + aplay pipeline started.", flush=True)

    def _stop_processes_unlocked(self):
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
        with self._lock:
            piper_dead = self.piper is None or self.piper.poll() is not None
            aplay_dead = self.aplay is None or self.aplay.poll() is not None
            if piper_dead or aplay_dead:
                print("[TTS] Pipeline dead restarting", flush=True)
                self._start_processes()

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
        self.enabled = False
        self.q.put(None)
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

    def _worker(self):
        while True:
            text = self.q.get()
            if text is None:
                break
            try:
                self._ensure_running()
                payload = (text + "\n").encode("utf-8")
                with self._lock:
                    if self.piper is None or self.piper.stdin is None:
                        raise RuntimeError("piper stdin not available")
                    self.piper.stdin.write(payload)
                    self.piper.stdin.flush()
                print(f"[TTS] Queued spoken: {text}", flush=True)
            except Exception as e:
                print(f"[TTS] Exception while speaking: {e}", flush=True)
                with self._lock:
                    self._stop_processes_unlocked()

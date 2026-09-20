#Claude sonnet 5 helped

import cv2
import time
import numpy as np
from tts_espeak import AsyncTTS
from ollama import Client
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import os
import signal
from pynput import keyboard


class ChaplinOutput(BaseModel):
    list_of_changes: str
    corrected_text: str


class Chaplin:

    # Kept as a class-level constant purely so it's easy to find/edit.
    # NOTE: the trailing "/no_think" hint is a Qwen3-specific convention
    # that suppresses its internal <think>...</think> reasoning trace.
    # On Orin Nano that reasoning trace is very often the single biggest
    # source of correction latency for reasoning-capable models. It's a
    # no-op (harmless) for non-Qwen3 models. See CHAPLIN_DISABLE_THINKING
    # below if you ever want to turn this off.
    CORRECTION_SYSTEM_PROMPT = (
        "You are an assistant that helps make corrections to the output of a "
        "lipreading model. The text you will receive was transcribed using a "
        "video-to-text system that attempts to lipread the subject speaking in "
        "the video, so the text will likely be imperfect. The input text will "
        "also be in all-caps, although your response should be capitalized "
        "correctly and should NOT be in all-caps.\n\n"
        "If something seems unusual, assume it was mistranscribed. Do your best "
        "to infer the words actually spoken, and make changes to the "
        "mistranscriptions in your response. Do not add more words or content, "
        "just change the ones that seem to be out of place (and, therefore, "
        "mistranscribed). Do not change even the wording of sentences, just "
        "individual words that look nonsensical in the context of all of the "
        "other words in the sentence.\n\n"
        "Also, add correct punctuation to the entire text. ALWAYS end each "
        "sentence with the appropriate sentence ending: '.', '?', or '!'.\n\n"
        "Return the corrected text in the format of 'list_of_changes' and "
        "'corrected_text'."
    )

    def __init__(self):
        self.vsr_model = None

        # set up text-to-speech
        self.tts = AsyncTTS()
        print(f"[TTS] Using ALSA device: {self.tts.alsa_device}", flush=True)

        # flag to toggle recording
        self.recording = False
        # flag to trigger clean exit from the capture loop
        self._shutdown = False

        # ------------------------------------------------------------------
        # Single-worker executor. This is the ENTIRE concurrency model now:
        # capture/record/display stays on the main thread (has to, it's
        # driving a live camera + a window), and every downstream stage --
        # VSR inference, Ollama correction, TTS, keyboard typing -- runs
        # back-to-back inside this one worker thread, one utterance at a
        # time, in submission order. Because there's only ever one worker,
        # ordering is guaranteed for free -- no asyncio, no Condition, no
        # sequence numbers required.
        # ------------------------------------------------------------------
        self.executor = ThreadPoolExecutor(max_workers=1)

        # video params - res_factor now from INI [capture] if present
        self.output_prefix = "webcam"
        self.res_factor = 2
        self.fps = 16
        self.frame_compression = 25
        ini_path = os.getenv("CHAPLIN_CONFIG", "./configs/LRS3_V_WER19.1.ini")
        try:
            import configparser
            cp = configparser.ConfigParser()
            cp.read(ini_path)
            if cp.has_section("capture"):
                self.res_factor = int(cp.get("capture", "res_factor", fallback=str(self.res_factor)))
                self.fps = int(cp.get("capture", "fps", fallback=str(self.fps)))
                self.frame_compression = int(cp.get("capture", "frame_compression", fallback=str(self.frame_compression)))
            if cp.has_section("performance"):
                os.environ.setdefault("MEDIAPIPE_DETECT_EVERY", cp.get("performance", "mediapipe_detect_every", fallback="8"))
                os.environ.setdefault("CHAPLIN_CAMERA_OPEN_RETRIES", cp.get("performance", "camera_open_retries", fallback="30"))
                os.environ.setdefault("CHAPLIN_CAMERA_OPEN_RETRY_DELAY", cp.get("performance", "camera_open_retry_delay", fallback="3.0"))
            if cp.has_section("tts"):
                os.environ.setdefault("TTS_ALSA_DEVICE", cp.get("tts", "alsa_device", fallback="default"))
                os.environ.setdefault("CHAPLIN_TTS_RAW", cp.get("tts", "tts_raw", fallback="1"))
                os.environ.setdefault("CHAPLIN_TTS_CORRECTED", cp.get("tts", "tts_corrected", fallback="0"))
                os.environ.setdefault("CHAPLIN_DISABLE_OLLAMA", cp.get("tts", "disable_ollama", fallback="1"))
            if cp.has_section("ollama"):
                os.environ.setdefault("OLLAMA_MODEL", cp.get("ollama", "model", fallback="qwen3:1.7b"))
            print(f"[Chaplin] Loaded startup config from {ini_path}: res_factor={self.res_factor} fps={self.fps}", flush=True)
        except Exception as e:
            print(f"[Chaplin] Could not read {ini_path}: {e}, using defaults res_factor={self.res_factor}", flush=True)
        self.frame_interval = 1 / self.fps

        # setup keyboard controller for typing
        self.kbd_controller = keyboard.Controller()

        # setup ollama client (SYNCHRONOUS -- see note above on why)
        ollama_host = os.getenv("OLLAMA_HOST")  # e.g. http://localhost:11434
        self.ollama_timeout_sec = float(os.getenv("OLLAMA_TIMEOUT_SEC", "20"))
        client_kwargs = {"timeout": self.ollama_timeout_sec}
        if ollama_host:
            client_kwargs["host"] = ollama_host
        self.ollama_client = Client(**client_kwargs)

        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:4b")
        self.ollama_num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "256"))

        disable_thinking = os.getenv("CHAPLIN_DISABLE_THINKING", "1").strip() == "1"
        self.correction_system_prompt = self.CORRECTION_SYSTEM_PROMPT
        if disable_thinking and "qwen3" in self.ollama_model.lower():
            self.correction_system_prompt += "\n\n/no_think"

        print(f"[Chaplin] Ollama host: {ollama_host or '(default)'}", flush=True)
        print(f"[Chaplin] Model: {self.ollama_model} "
              f"(timeout={self.ollama_timeout_sec}s, num_predict={self.ollama_num_predict})",
              flush=True)

        # setup global hotkey for toggling recording with option/alt key
        self.hotkey = keyboard.GlobalHotKeys({
            '<alt>': self.toggle_recording
        })
        self.hotkey.start()

        # register signal handlers so Ctrl-C and SIGTERM both trigger clean shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        sig_name = "SIGINT (Ctrl-C)" if signum == signal.SIGINT else "SIGTERM"
        print(f"\n[Chaplin] {sig_name} received — shutting down cleanly...", flush=True)
        self._shutdown = True
        self.recording = False  # stop any active recording

    def toggle_recording(self):
        # toggle recording when alt/option key is pressed
        self.recording = not self.recording

    # ----------------------------------------------------------------------
    # Correction / TTS / typing -- fully synchronous, called from inside
    # perform_inference() which itself only ever runs on the single
    # executor worker thread. No locking needed: there is structurally
    # only ever one of these in flight at a time.
    # ----------------------------------------------------------------------
    def correct_output(self, output):
        # Skip LLM entirely for real-time wearable - env CHAPLIN_DISABLE_OLLAMA=1
        if os.getenv("CHAPLIN_DISABLE_OLLAMA", "0").strip() == "1":
            corrected = output.strip().capitalize()
            if corrected and corrected[-1] not in ('.', '?', '!'):
                corrected += '.'
            corrected += ' '
            print(f"[Chaplin] CHAPLIN_DISABLE_OLLAMA=1 - skipping LLM, using raw: {corrected.strip()}", flush=True)
            return corrected

        t0 = time.perf_counter()
        corrected_text = None

        try:
            response = self.ollama_client.chat(
                model=self.ollama_model,
                messages=[
                    {
                        'role': 'system',
                        'content': self.correction_system_prompt,
                    },
                    {
                        'role': 'user',
                        'content': f"Transcription:\n\n{output}",
                    },
                ],
                format=ChaplinOutput.model_json_schema(),
                options={"num_predict": self.ollama_num_predict},
            )
            chat_output = ChaplinOutput.model_validate_json(
                response['message']['content'])
            corrected_text = chat_output.corrected_text.strip()
            if not corrected_text:
                raise ValueError("empty corrected_text returned by ollama")

        except Exception as e:
            # Covers timeouts (httpx.TimeoutException), connection errors,
            # malformed JSON, empty responses -- anything. A slow/broken
            # Ollama call must NEVER be allowed to silently drop an
            # utterance on a wearable device; fall back to the raw VSR
            # output instead of skipping it.
            import traceback
            print(f"[Chaplin] WARNING: ollama correction failed/timed out: {e}", flush=True)
            traceback.print_exc()
            corrected_text = output.strip().capitalize()

        if corrected_text and corrected_text[-1] not in ('.', '?', '!'):
            corrected_text += '.'
        corrected_text += ' '

        t1 = time.perf_counter()
        print(f"[PERF] correct_output: {(t1 - t0) * 1000:.1f}ms", flush=True)
        return corrected_text

    def perform_inference(self, video_path):
        """
        Runs entirely inside the single-worker executor thread.
        VSR -> (optional immediate RAW TTS) -> correction -> typing, strictly sequential.
        Set CHAPLIN_TTS_RAW=1 (default) for real-time audio without waiting for LLM.
        Set CHAPLIN_TTS_CORRECTED=0 (default) to skip speaking corrected version.
        """
        t0 = time.perf_counter()
        output = self.vsr_model(video_path)
        t1 = time.perf_counter()
        print(f"[PERF] VSR inference: {(t1 - t0) * 1000:.1f}ms | {video_path}", flush=True)
        print(f"\n\033[48;5;21m\033[97m\033[1m RAW OUTPUT \033[0m: {output}\n", flush=True)

        # REAL-TIME AUDIO: speak raw immediately, don't wait 20s for LLM
        tts_raw = os.getenv("CHAPLIN_TTS_RAW", "1").strip() == "1"
        tts_corrected = os.getenv("CHAPLIN_TTS_CORRECTED", "0").strip() == "1"

        if tts_raw:
            try:
                raw_spaced = output.strip().capitalize() + ". "
                self.tts.say(raw_spaced)
            except Exception as e:
                print(f"[Chaplin] ERROR: tts.say(raw) failed: {e}", flush=True)

        corrected_text = self.correct_output(output)
        t2 = time.perf_counter()

        if tts_corrected:
            try:
                self.tts.say(corrected_text)
            except Exception as e:
                print(f"[Chaplin] ERROR: tts.say(corrected) failed: {e}", flush=True)

        try:
            self.kbd_controller.type(corrected_text)
        except Exception as e:
            print(f"[Chaplin] ERROR: kbd_controller.type() failed: {e}", flush=True)

        t3 = time.perf_counter()
        print(
            f"[PERF] vsr={(t1 - t0) * 1000:.0f}ms "
            f"correction={(t2 - t1) * 1000:.0f}ms "
            f"tts+type={(t3 - t2) * 1000:.0f}ms "
            f"TOTAL={(t3 - t0) * 1000:.0f}ms",
            flush=True,
        )

        return {
            "output": corrected_text,
            "video_path": video_path,
        }

    # ------------------------------------------------------------------
    # Everything below (GStreamer pipeline construction, camera open,
    # stale-clip cleanup, main capture loop) is UNCHANGED from your
    # existing implementation -- only the async/typing-lock plumbing in
    # __init__ / correct_output / perform_inference / cleanup was removed.
    # ------------------------------------------------------------------

    def _build_gstreamer_pipeline(self, width, height, fps, stream=False,
                                   stream_host=None, stream_port=None,
                                   sensor_width=1280, sensor_height=720,
                                   stream_bitrate=None):
        """
        Build a hardware-accelerated GStreamer pipeline for the IMX519 on Jetson.
        FIX: Always request a VALID sensor mode (1280x720) from nvarguscamerasrc,
        then scale down to inference size (212x160) with nvvidconv. Requesting
        212x160 directly from nvarguscamerasrc is invalid and causes
        NvBufSurfaceFromFd Failed on JP6.
        """
        stream_width = int(os.getenv("STREAM_WIDTH", "640"))
        stream_height = int(os.getenv("STREAM_HEIGHT", "480"))
        stream_bitrate = stream_bitrate or int(os.getenv("STREAM_BITRATE", "1500000"))

        if not stream:
            return (
                f"nvarguscamerasrc sensor-id=0 sensor-mode=3 ! "
                f"video/x-raw(memory:NVMM),width={sensor_width},height={sensor_height},"
                f"framerate={fps}/1,format=NV12 ! "
                f"nvvidconv ! video/x-raw,format=BGRx,width={width},height={height} ! "
                f"videoconvert ! video/x-raw,format=BGR ! "
                f"appsink drop=1 max-buffers=2"
            )

        return (
            f"nvarguscamerasrc sensor-id=0 sensor-mode=3 ! "
            f"video/x-raw(memory:NVMM),width={sensor_width},height={sensor_height},"
            f"framerate={fps}/1,format=NV12 ! "
            f"tee name=t "
            f"t. ! queue leaky=downstream max-size-buffers=2 ! "
            f"nvvidconv ! video/x-raw,format=BGRx,width={width},height={height} ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=1 max-buffers=2 "
            f"t. ! queue leaky=downstream max-size-buffers=2 ! "
            f"nvvidconv ! video/x-raw(memory:NVMM),width={stream_width},height={stream_height} ! "
            f"nvv4l2h264enc bitrate={stream_bitrate} insert-sps-pps=1 ! "
            f"h264parse ! rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={stream_host} port={stream_port} sync=false async=false"
        )

    def _warm_ollama_model(self):
        """
        Load the correction model into memory. Deliberately called AFTER
        _open_camera() succeeds (see start_webcam()) -- Argus allocates its
        VI/CSI hardware capture buffers once, at CaptureSession init time,
        and holds them for the life of the session. Warming a multi-GB
        Ollama model into this device's shared 8GB memory BEFORE that
        allocation happens can starve it, producing NvMap
        "Error InsufficientMemory" / "No cameras available" failures.
        Once the camera session is live, its buffers are already reserved,
        so it's safe to let Ollama consume memory afterward.
        """
        t0 = time.perf_counter()
        try:
            self.ollama_client.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': ''}],
                options={"num_predict": 1},
             )
            t1 = time.perf_counter()
            print(f"[Chaplin] Ollama model '{self.ollama_model}' warmed "
                  f"({(t1 - t0):.1f}s).", flush=True)
        except Exception as e:
            print(f"[Chaplin] WARNING: Ollama warm-up failed (will load "
                  f"lazily on first correction instead): {e}", flush=True)

    def _open_camera(self):
        """
        Open the camera using either hardware-accelerated GStreamer (IMX519 on
        Jetson) or a plain USB webcam, controlled by the USE_HW_CAMERA env var.

        Set USE_HW_CAMERA=1 in the container/shell environment to use the IMX519.
        Leave it unset (or set to 0) to use a standard USB webcam via V4L2.
        """
        cap_width = (640 // self.res_factor) & ~1   # round down to even -> 212
        cap_height = (480 // self.res_factor) & ~1  # round down to even -> 160

        use_hw = os.getenv("USE_HW_CAMERA", "0").strip() == "1"
        stream = os.getenv("CHAPLIN_STREAM", "0").strip() == "1"
        stream_host = os.getenv("STREAM_HOST", "192.168.1.17")
        stream_port = os.getenv("STREAM_PORT", "5000")

        # Free GPU memory before Argus alloc on 8GB Nano
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        if use_hw:
            gst_pipeline = self._build_gstreamer_pipeline(
                cap_width, cap_height, self.fps,
                stream=stream, stream_host=stream_host, stream_port=stream_port)
            print(f"[Chaplin] USE_HW_CAMERA=1 — opening IMX519 via GStreamer:", flush=True)
            if stream:
                print(f"[Chaplin]   Streaming live H264 to {stream_host}:{stream_port}", flush=True)
            print(f"[Chaplin]   {gst_pipeline}", flush=True)

            # cap.isOpened() lies here -- it can return True even when the
            # underlying Argus CaptureSession failed to create. The only
            # trustworthy check is an actual frame read.
            #
            # Retry budget is generous and NOT a short transient-race
            # workaround: Argus has been observed needing anywhere from
            # ~5s to 40s+ to settle after a fresh nvargus-daemon/container
            # start before it can create a capture session, and on at least
            # one occasion did not succeed even after 60s. There is no
            # known reliable upper bound on this hardware, so we retry for
            # up to several minutes with visible progress rather than
            # giving up early.
            #
            # NOTE: this deliberately does NOT fall back to
            # cv2.VideoCapture(0) on failure. On this hardware /dev/video0
            # IS the IMX519's raw V4L2 sensor node (confirmed via
            # `v4l2-ctl --list-devices` -- there is no separate USB
            # webcam), so that "fallback" can never produce real frames;
            # it just spins forever on select() timeouts while looking
            # like progress. Set CHAPLIN_CAMERA_ALLOW_V4L2_FALLBACK=1 to
            # re-enable it if a real second camera is ever added.
            max_attempts = int(os.getenv("CHAPLIN_CAMERA_OPEN_RETRIES", "10"))
            retry_delay = float(os.getenv("CHAPLIN_CAMERA_OPEN_RETRY_DELAY", "2.0"))
            got_frame = False

            for attempt in range(1, max_attempts + 1):
                cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        got_frame = True
                        break
                    if attempt == 1 or attempt % 5 == 0:
                        print(f"[Chaplin] Camera opened but no frame yet "
                              f"(attempt {attempt}/{max_attempts}, "
                              f"~{attempt * retry_delay:.0f}s elapsed) — "
                              f"Argus may still be settling, retrying...", flush=True)
                else:
                    if attempt == 1 or attempt % 5 == 0:
                        print(f"[Chaplin] cv2.VideoCapture failed to open "
                              f"(attempt {attempt}/{max_attempts}, "
                              f"~{attempt * retry_delay:.0f}s elapsed) — retrying...", flush=True)
                cap.release()
                if attempt < max_attempts:
                     time.sleep(retry_delay)

            if not got_frame:
                allow_v4l2_fallback = os.getenv(
                    "CHAPLIN_CAMERA_ALLOW_V4L2_FALLBACK", "0").strip() == "1"
                if allow_v4l2_fallback:
                    print(f"[Chaplin] WARNING: GStreamer pipeline never produced "
                          f"a real frame after {max_attempts} attempts "
                          f"(~{max_attempts * retry_delay:.0f}s) — falling back to "
                          f"cv2.VideoCapture(0).", flush=True)
                    use_hw = False
                else:
                    raise RuntimeError(
                        f"[Chaplin] FATAL: Argus never produced a real camera "
                        f"frame after {max_attempts} attempts "
                        f"(~{max_attempts * retry_delay:.0f}s), even though the host-side "
                        f"preflight check (see start_chaplin_container.sh) passed moments "
                        f"earlier. This strongly suggests something re-acquired the camera "
                        f"in between -- check for a second chaplin container/process, or a "
                        f"race with another script. Check `sudo lsof /dev/video0` and "
                        f"`sudo docker ps -a` right now, while this error is fresh."
                    )

        if not use_hw:
            print("[Chaplin] Opening USB webcam via cv2.VideoCapture(0).", flush=True)
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Chaplin] Camera ready: {frame_width}x{frame_height} @ {self.fps}fps", flush=True)
        return cap, frame_width, frame_height

    def _cleanup_stale_clips(self):
        """Remove any .mp4 clips left over from a previous crashed session."""
        stale = [f for f in os.listdir()
                 if f.startswith(self.output_prefix) and f.endswith('.mp4')]
        if stale:
            print(f"[Chaplin] Removing {len(stale)} stale clip(s) from previous session...",
                  flush=True)
            for f in stale:
                try:
                    os.remove(f)
                except OSError:
                    pass

    def start_webcam_with_existing_cap(self, cap, frame_width, frame_height):
        """Same as start_webcam but reuses already-opened camera to avoid Argus OOM on 8GB"""
        self._cleanup_stale_clips()
        print(f"[Chaplin] Reusing existing camera: {frame_width}x{frame_height} @ {self.fps}fps", flush=True)
        # Now safe to warm Ollama after camera buffers reserved
        self._warm_ollama_model()
        self.tts.say("Chaplin is ready.")
        self.tts.say("Chaplin is ready.")
        cv2.namedWindow('Chaplin', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Chaplin', 640, 480)
        last_frame_time = time.time()
        futures = []
        output_path = ""
        out = None
        frame_count = 0
        try:
            while not self._shutdown:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[Chaplin] 'q' pressed — exiting.", flush=True)
                    break
                current_time = time.time()
                if current_time - last_frame_time >= self.frame_interval:
                    ret, frame = cap.read()
                    if ret:
                        compressed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if self.recording:
                            if out is None:
                                output_path = self.output_prefix + str(time.time_ns() // 1_000_000) + '.mp4'
                                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (frame_width, frame_height), False)
                            out.write(compressed_frame)
                            last_frame_time = current_time
                            cv2.circle(compressed_frame, (frame_width - 20, 20), 10, (0, 0, 0), -1)
                            frame_count += 1
                        elif not self.recording and frame_count > 0:
                            if out is not None:
                                out.release()
                            if frame_count >= self.fps * 2:
                                futures.append(self.executor.submit(self.perform_inference, output_path))
                            else:
                                os.remove(output_path)
                            output_path = self.output_prefix + str(time.time_ns() // 1_000_000) + '.mp4'
                            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (frame_width, frame_height), False)
                            frame_count = 0
                        display_frame = cv2.resize(cv2.flip(compressed_frame, 1), (640, 480))
                        h, w = display_frame.shape[:2]
                        # Guide box
                        box_x1, box_y1 = w//4, h//4
                        box_x2, box_y2 = 3*w//4, 3*h//4
                        cv2.rectangle(display_frame, (box_x1, box_y1), (box_x2, box_y2), (0,255,0), 2)
                        cv2.putText(display_frame, "Fit face here", (box_x1, box_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        cv2.line(display_frame, (w//2-20, h//2), (w//2+20, h//2), (0,255,0), 1)
                        cv2.line(display_frame, (w//2, h//2-20), (w//2, h//2+20), (0,255,0), 1)
                        # Lighting feedback
                        try:
                            roi = display_frame[box_y1:box_y2, box_x1:box_x2]
                            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape)==3 else roi
                            mean_b = float(np.mean(gray_roi))
                            if mean_b < 50:
                                txt, col = f"Lighting BAD dark {mean_b:.0f} - add light", (0,0,255)
                            elif mean_b > 200:
                                txt, col = f"Lighting BAD bright {mean_b:.0f}", (0,0,255)
                            else:
                                txt, col = f"Lighting OK {mean_b:.0f}", (0,255,0)
                            cv2.putText(display_frame, txt, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                        except Exception:
                            pass
                        cv2.imshow('Chaplin', display_frame)
                for fut in futures:
                    if fut.done():
                        result = fut.result()
                        os.remove(result["video_path"])
                        futures.remove(fut)
                    else:
                        break
        finally:
            print("[Chaplin] Cleaning up...", flush=True)
            if out is not None:
                out.release()
            if output_path and os.path.exists(output_path):
                print(f"[Chaplin] Discarding partial clip: {output_path}", flush=True)
                os.remove(output_path)
            cap.release()
            cv2.destroyAllWindows()
            self._cleanup_stale_clips()
            for fut in futures:
                fut.cancel()
            self.hotkey.stop()
            self.executor.shutdown(wait=False)
            print("[Chaplin] Shutdown complete.", flush=True)

    def start_webcam(self):
        # remove any clips left over from a previous crashed session
        self._cleanup_stale_clips()

        cap, frame_width, frame_height = self._open_camera()
        # Camera capture session is confirmed live and its hardware buffers
        # are already allocated -- safe to let Ollama consume memory now.
        self._warm_ollama_model()

        self.tts.say("Chaplin is ready.")


        # Speak the readiness greeting only now, after the camera is
        # confirmed open -- not in __init__. piper's synthesis is CPU-bound
        # (onnxruntime) and was found to intermittently starve
        # nvargus-daemon (host-side, sharing the same physical cores) of
        # the scheduling time it needs to service an Argus capture-session
        # RPC promptly, causing camera open to take anywhere from ~5s to
        # 60s+ unpredictably. Speaking after camera-open removes any chance
        # of that overlap entirely, rather than just capping piper's thread
        # count (see tts_espeak.py) which reduces but doesn't guarantee
        # zero contention.
        self.tts.say("Chaplin is ready.")

        cv2.namedWindow('Chaplin', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Chaplin', 640, 480)

        last_frame_time = time.time()
        futures = []
        output_path = ""
        out = None
        frame_count = 0

        try:
            while not self._shutdown:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[Chaplin] 'q' pressed — exiting.", flush=True)
                    break

                current_time = time.time()

                # conditional ensures that the video is recorded at the correct frame rate
                if current_time - last_frame_time >= self.frame_interval:
                    ret, frame = cap.read()
                    if ret:
                        # NOTE: this JPEG encode/decode round-trip is CPU work
                        # done every single frame at self.fps. It's used here
                        # purely to (a) force grayscale and (b) apply a light
                        # compression pass before the frame ever hits disk.
                        # If you're chasing capture-loop latency on Orin Nano,
                        # replacing this with a plain
                        # cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) is
                        # meaningfully cheaper and produces the same
                        # grayscale-only VideoWriter input, since the mp4v
                        # VideoWriter compresses on its own anyway.
                        #encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.frame_compression]
                        #_, buffer = cv2.imencode('.jpg', frame, encode_param)
                        #compressed_frame = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
                        compressed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        if self.recording:
                            if out is None:
                                output_path = self.output_prefix + \
                                    str(time.time_ns() // 1_000_000) + '.mp4'
                                out = cv2.VideoWriter(
                                    output_path,
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    self.fps,
                                    (frame_width, frame_height),
                                    False  # isColor
                                )
                            out.write(compressed_frame)
                            last_frame_time = current_time

                            # circle to indicate recording, only appears in the
                            # window and is not present in video saved to disk
                            cv2.circle(compressed_frame,
                                       (frame_width - 20, 20), 10, (0, 0, 0), -1)
                            frame_count += 1

                        # check if not recording AND video is at least 2 seconds long
                        elif not self.recording and frame_count > 0:
                            if out is not None:
                                out.release()

                            # only run inference if the video is at least 2 seconds long
                            if frame_count >= self.fps * 2:
                                futures.append(self.executor.submit(
                                    self.perform_inference, output_path))
                            else:
                                os.remove(output_path)

                            output_path = self.output_prefix + \
                                str(time.time_ns() // 1_000_000) + '.mp4'
                            out = cv2.VideoWriter(
                                output_path,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                self.fps,
                                (frame_width, frame_height),
                                False  # isColor
                            )
                            frame_count = 0

                        # display the frame in the window with face guide overlay + lighting
                        display_frame = cv2.resize(cv2.flip(compressed_frame, 1), (640, 480))
                        h, w = display_frame.shape[:2]
                        box_x1, box_y1 = w//4, h//4
                        box_x2, box_y2 = 3*w//4, 3*h//4
                        cv2.rectangle(display_frame, (box_x1, box_y1), (box_x2, box_y2), (0,255,0), 2)
                        cv2.putText(display_frame, "Fit face here", (box_x1, box_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        cv2.line(display_frame, (w//2-20, h//2), (w//2+20, h//2), (0,255,0), 1)
                        cv2.line(display_frame, (w//2, h//2-20), (w//2, h//2+20), (0,255,0), 1)
                        try:
                            roi = display_frame[box_y1:box_y2, box_x1:box_x2]
                            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape)==3 else roi
                            mean_b = float(np.mean(gray_roi))
                            if mean_b < 50:
                                txt, col = f"Lighting BAD dark {mean_b:.0f} - add light", (0,0,255)
                            elif mean_b > 200:
                                txt, col = f"Lighting BAD bright {mean_b:.0f}", (0,0,255)
                            else:
                                txt, col = f"Lighting OK {mean_b:.0f}", (0,255,0)
                            cv2.putText(display_frame, txt, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                        except Exception:
                            pass
                        cv2.imshow('Chaplin', display_frame)
                # ensures that videos are handled in the order they were recorded
                for fut in futures:
                    if fut.done():
                        result = fut.result()
                        # once done processing, delete the video with the video path
                        os.remove(result["video_path"])
                        futures.remove(fut)
                    else:
                        break

        finally:
            print("[Chaplin] Cleaning up...", flush=True)

            # release active VideoWriter — discard the partial clip
            if out is not None:
                out.release()
            if output_path and os.path.exists(output_path):
                print(f"[Chaplin] Discarding partial clip: {output_path}", flush=True)
                os.remove(output_path)

            # release camera
            cap.release()
            cv2.destroyAllWindows()

            # clean up any remaining stale clips
            self._cleanup_stale_clips()

            # cancel any pending inference futures
            for fut in futures:
                fut.cancel()

            # stop global hotkey listener
            self.hotkey.stop()

            # shutdown executor
            self.executor.shutdown(wait=False)

            print("[Chaplin] Shutdown complete.", flush=True)

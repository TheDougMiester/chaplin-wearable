import cv2
import time
from tts_espeak import AsyncTTS
from ollama import AsyncClient
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import os
import signal
from pynput import keyboard
import asyncio


class ChaplinOutput(BaseModel):
    list_of_changes: str
    corrected_text: str


class Chaplin:
    def __init__(self):
        self.vsr_model = None
        # set up text-to-speech
        self.tts = AsyncTTS()
        print(f"[TTS] Using ALSA device: {self.tts.alsa_device}", flush=True)
        # flag to toggle recording
        self.recording = False
        # flag to trigger clean exit from the capture loop
        self._shutdown = False

        # thread stuff
        self.executor = ThreadPoolExecutor(max_workers=1)

        # video params
        self.output_prefix = "webcam"
        self.res_factor = 3
        self.fps = 16
        self.frame_interval = 1 / self.fps
        self.frame_compression = 25

        # setup keyboard controller for typing
        self.kbd_controller = keyboard.Controller()

        # setup async ollama client
        self.ollama_client = AsyncClient()
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:4b")
        print(f"[Chaplin] Ollama endpoint: {self.ollama_client}", flush=True)
        print(f"[Chaplin] Model: {self.ollama_model}", flush=True)

        # setup asyncio event loop in background thread
        self.loop = asyncio.new_event_loop()
        self.async_thread = ThreadPoolExecutor(max_workers=1)
        self.async_thread.submit(self._run_event_loop)

        # sequence tracking to ensure outputs are typed in order
        self.next_sequence_to_type = 0
        self.current_sequence = 0  # counter for assigning sequence numbers
        self.typing_lock = None    # will be created in async loop
        self._init_async_resources()

        # setup global hotkey for toggling recording with option/alt key
        self.hotkey = keyboard.GlobalHotKeys({
            '<alt>': self.toggle_recording
        })
        self.hotkey.start()

        # register signal handlers so Ctrl-C and SIGTERM both trigger clean shutdown
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        sig_name = "SIGINT (Ctrl-C)" if signum == signal.SIGINT else "SIGTERM"
        print(f"\n[Chaplin] {sig_name} received — shutting down cleanly...", flush=True)
        self._shutdown = True
        self.recording = False  # stop any active recording

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _init_async_resources(self):
        """Initialize async resources in the async loop"""
        future = asyncio.run_coroutine_threadsafe(
            self._create_async_lock(), self.loop)
        future.result()  # wait for it to complete

    async def _create_async_lock(self):
        """Create asyncio.Lock and Condition in the event loop's context"""
        self.typing_lock = asyncio.Lock()
        self.typing_condition = asyncio.Condition(self.typing_lock)

    def toggle_recording(self):
        # toggle recording when alt/option key is pressed
        self.recording = not self.recording

    async def correct_output_async(self, output, sequence_num):
        try:
            return await self._correct_output_async_inner(output, sequence_num)
        except Exception as e:
            import traceback
            print(f"[Chaplin] ERROR: correct_output_async failed for sequence {sequence_num}: {e}", flush=True)
            traceback.print_exc()
            # make sure a failed task doesn't permanently stall the typing/tts
            # sequence for every task queued behind it
            async with self.typing_condition:
                if self.next_sequence_to_type == sequence_num:
                    self.next_sequence_to_type += 1
                    self.typing_condition.notify_all()
            return ""

    async def _correct_output_async_inner(self, output, sequence_num):
        # perform inference on the raw output to get back a "correct" version
        response = await self.ollama_client.chat(
            model=self.ollama_model,
            messages=[
                {
                    'role': 'system',
                    'content': (
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
                },
                {
                    'role': 'user',
                    'content': f"Transcription:\n\n{output}"
                }
            ],
            format=ChaplinOutput.model_json_schema()
        )

        # get only the corrected text
        chat_output = ChaplinOutput.model_validate_json(
            response['message']['content'])

        # if last character isn't a sentence ending (happens sometimes), add a period
        chat_output.corrected_text = chat_output.corrected_text.strip()
        if not chat_output.corrected_text:
            print(f"[Chaplin] WARNING: empty corrected_text for sequence {sequence_num}, skipping.", flush=True)
            async with self.typing_condition:
                while self.next_sequence_to_type != sequence_num:
                    await self.typing_condition.wait()
                self.next_sequence_to_type += 1
                self.typing_condition.notify_all()
            return ""

        if chat_output.corrected_text[-1] not in ['.', '?', '!']:
            chat_output.corrected_text += '.'

        # add space at the end
        chat_output.corrected_text += ' '

        # wait until it's this task's turn to type
        async with self.typing_condition:
            while self.next_sequence_to_type != sequence_num:
                await self.typing_condition.wait()

            # give text to speech output FIRST so a typing failure can never
            # suppress audio output
            try:
                self.tts.say(chat_output.corrected_text)
            except Exception as e:
                print(f"[Chaplin] ERROR: tts.say() failed: {e}", flush=True)

            # this task's turn to type the corrected text
            try:
                self.kbd_controller.type(chat_output.corrected_text)
            except Exception as e:
                print(f"[Chaplin] ERROR: kbd_controller.type() failed: {e}", flush=True)

            # increment sequence and notify next task
            self.next_sequence_to_type += 1
            self.typing_condition.notify_all()

        return chat_output.corrected_text

    def perform_inference(self, video_path):
        import time
        t0 = time.perf_counter()
        output = self.vsr_model(video_path)
        t1 = time.perf_counter()
        print(f"[PERF] perform_inference TOTAL: {(t1-t0)*1000:.1f}ms | {video_path}", flush=True)
        print(f"\n\033[48;5;21m\033[97m\033[1m RAW OUTPUT \033[0m: {output}\n", flush=True)
        sequence_num = self.current_sequence
        self.current_sequence += 1
        asyncio.run_coroutine_threadsafe(
            self.correct_output_async(output, sequence_num),
            self.loop
        )
        return {
            "output": output,
            "video_path": video_path
        }
    def _build_gstreamer_pipeline(self, width, height, fps, stream=False,
                                   stream_host=None, stream_port=None,
                                   sensor_width=1280, sensor_height=720,
                                   stream_bitrate=None):
        """
        Build a hardware-accelerated GStreamer pipeline for the IMX519 on Jetson.
        Requires the following environment variables to be set before launch:
            EGL_PLATFORM=device
            EGL_DEVICE_ID=/dev/dri/renderD128
        Both DISPLAY and WAYLAND_DISPLAY must be unset.
        See run_chaplin_workspace_x11.sh for how these are configured.

        If stream=True, the single nvarguscamerasrc capture is tee'd into two
        branches so we don't open a second, competing Argus capture session:
          - inference branch: scaled down to (width, height) -> appsink, same
            as before, feeds cv2.VideoCapture for the VSR pipeline.
          - streaming branch: scaled down to STREAM_WIDTH/STREAM_HEIGHT ->
            nvv4l2h264enc -> rtph264pay -> udpsink, sent live to
            stream_host:stream_port (see run_chaplin_workspace_x11.sh).

        sensor_width/sensor_height (the nvarguscamerasrc capture caps) stay
        fixed at 1280x720 -- that's the smallest of the IMX519's actual
        discrete Argus sensor modes (4656x3496 / 3840x2160 / 1920x1080 /
        1280x720), so it's the only sane choice for the raw capture request;
        arbitrary resolutions like 640x480 aren't valid sensor modes and
        risk failing caps negotiation entirely. Instead, the STREAMING
        branch gets its own nvvidconv scale-down AFTER capture, so the
        H264 encoder's buffer pool is smaller (real, previously-untested
        memory pressure -- caused an actual OOM kill on an 8GB device
        already running a warm Ollama model) without touching sensor
        negotiation at all.
        """
        stream_width = int(os.getenv("STREAM_WIDTH", "640"))
        stream_height = int(os.getenv("STREAM_HEIGHT", "480"))
        stream_bitrate = stream_bitrate or int(os.getenv("STREAM_BITRATE", "1500000"))

        if not stream:
            return (
                f"nvarguscamerasrc sensor_id=0 ! "
                f"video/x-raw(memory:NVMM),width={width},height={height},"
                f"framerate={fps}/1,format=NV12 ! "
                f"nvvidconv ! "
                f"video/x-raw,format=BGRx ! "
                f"videoconvert ! "
                f"video/x-raw,format=BGR ! "
                f"appsink drop=1 max-buffers=2"
            )

        return (
            f"nvarguscamerasrc sensor_id=0 ! "
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

    def _open_camera(self):
        """
        Open the camera using either hardware-accelerated GStreamer (IMX519 on
        Jetson) or a plain USB webcam, controlled by the USE_HW_CAMERA env var.

        Set USE_HW_CAMERA=1 in the container/shell environment to use the IMX519.
        Leave it unset (or set to 0) to use a standard USB webcam via V4L2.
        """
        cap_width  = (640 // self.res_factor) & ~1   # round down to even -> 212
        cap_height = (480 // self.res_factor) & ~1   # round down to even -> 160
        use_hw = os.getenv("USE_HW_CAMERA", "0").strip() == "1"

        stream = os.getenv("CHAPLIN_STREAM", "0").strip() == "1"
        stream_host = os.getenv("STREAM_HOST", "192.168.1.17")
        stream_port = os.getenv("STREAM_PORT", "5000")

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
            max_attempts = int(os.getenv("CHAPLIN_CAMERA_OPEN_RETRIES", "20"))
            retry_delay = float(os.getenv("CHAPLIN_CAMERA_OPEN_RETRY_DELAY", "3.0"))
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
                        f"(~{max_attempts * retry_delay:.0f}s). Not falling back "
                        f"to cv2.VideoCapture(0) -- on this hardware that's the "
                        f"same IMX519 sensor via raw V4L2, which cannot produce "
                        f"usable frames and will just hang on select() timeouts. "
                        f"Check `sudo systemctl status nvargus-daemon` and "
                        f"`sudo docker ps -a` for a stale/competing session."
                    )

        if not use_hw:
            print("[Chaplin] Opening USB webcam via cv2.VideoCapture(0).", flush=True)
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cap_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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

    def start_webcam(self):
        # remove any clips left over from a previous crashed session
        self._cleanup_stale_clips()

        cap, frame_width, frame_height = self._open_camera()
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
                        # frame compression
                        encode_param = [
                            int(cv2.IMWRITE_JPEG_QUALITY), self.frame_compression]
                        _, buffer = cv2.imencode('.jpg', frame, encode_param)
                        compressed_frame = cv2.imdecode(
                            buffer, cv2.IMREAD_GRAYSCALE)

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

                        # display the frame in the window
                        #cv2.imshow('Chaplin', cv2.flip(compressed_frame, 1))
                        display_frame = cv2.resize(cv2.flip(compressed_frame, 1), (640, 480))
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

            # stop async event loop
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.async_thread.shutdown(wait=True)

            # shutdown executor
            self.executor.shutdown(wait=False)

            print("[Chaplin] Shutdown complete.", flush=True)

#!/usr/bin/env python3
"""
Record personal lip-reading dataset for fine-tuning to lower WER
Usage inside container:
  python3 record_personal_dataset.py --prompts prompts.txt --out_dir ~/wearer_data
prompts.txt example:
  I need water
  Vote early vote often
  Thank you
  I love you

Shows green Fit face here box + Lighting OK, records 3s per prompt when you press Alt.
Saves clip_001.mp4 + clip_001.txt
"""
import cv2, os, time, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prompts", default="prompts.txt")
parser.add_argument("--out_dir", default=os.path.expanduser("~/wearer_data"))
parser.add_argument("--res_factor", type=int, default=2)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# Load prompts
if os.path.exists(args.prompts):
    with open(args.prompts) as f:
        prompts = [l.strip() for l in f if l.strip()]
else:
    prompts = ["I need water", "Vote early vote often", "Thank you", "I love you", "How are you", "Good morning"]

print(f"Prompts: {prompts}")
print(f"Output dir: {args.out_dir}")
print("Press Alt to start/stop recording each prompt. Press q to quit.")

# Camera pipeline same as Chaplin
cap_width = (640 // args.res_factor) & ~1
cap_height = (480 // args.res_factor) & ~1
os.environ['EGL_PLATFORM']='device'
os.environ['EGL_DEVICE_ID']='/dev/dri/renderD128'
gst = f"nvarguscamerasrc sensor-id=0 sensor-mode=3 ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=16/1,format=NV12 ! nvvidconv ! video/x-raw,format=BGRx,width={cap_width},height={cap_height} ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2"
cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Camera failed, trying V4L2")
    cap = cv2.VideoCapture(0)

from pynput import keyboard
recording = False
out = None
frame_count = 0
current_prompt_idx = 0
output_path = ""

def toggle_recording():
    global recording, out, output_path, frame_count, current_prompt_idx
    recording = not recording
    if recording:
        print(f"\n[REC] Recording prompt {current_prompt_idx+1}/{len(prompts)}: {prompts[current_prompt_idx]}")
        output_path = os.path.join(args.out_dir, f"clip_{current_prompt_idx+1:03d}.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 16, (cap_width, cap_height), False)
        frame_count = 0
    else:
        if out is not None:
            out.release()
            print(f"[REC] Saved {output_path} ({frame_count} frames)")
            # Save transcript
            txt_path = os.path.splitext(output_path)[0] + ".txt"
            with open(txt_path, "w") as tf:
                tf.write(prompts[current_prompt_idx])
            print(f"[REC] Saved {txt_path}: {prompts[current_prompt_idx]}")
            current_prompt_idx += 1
            if current_prompt_idx >= len(prompts):
                print("All prompts recorded! Done.")
                return False
            print(f"Next prompt: {prompts[current_prompt_idx]} - Press Alt to record")
        out = None
        frame_count = 0

hotkey = keyboard.GlobalHotKeys({'<alt>': toggle_recording})
hotkey.start()

cv2.namedWindow('Personal Recorder', cv2.WINDOW_NORMAL)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        display = cv2.resize(cv2.flip(frame, 1), (640, 480))
        h,w = display.shape[:2]
        box_x1, box_y1 = w//4, h//4
        box_x2, box_y2 = 3*w//4, 3*h//4
        cv2.rectangle(display, (box_x1, box_y1), (box_x2, box_y2), (0,255,0), 2)
        if current_prompt_idx < len(prompts):
            cv2.putText(display, f"{current_prompt_idx+1}/{len(prompts)}: {prompts[current_prompt_idx]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # Lighting
        try:
            import numpy as np
            roi = display[box_y1:box_y2, box_x1:box_x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape)==3 else roi
            mean_b = float(np.mean(gray))
            col = (0,255,0) if 50 < mean_b < 200 else (0,0,255)
            cv2.putText(display, f"Lighting {mean_b:.0f}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        except:
            pass
        if recording:
            cv2.circle(display, (w-20, 20), 10, (0,0,255), -1)
            if out is not None:
                out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                frame_count += 1
        cv2.imshow('Personal Recorder', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if current_prompt_idx >= len(prompts):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    hotkey.stop()
    print(f"Dataset saved to {args.out_dir}")

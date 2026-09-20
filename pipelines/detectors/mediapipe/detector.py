#! /usr/bin/env python
# Fixed for mediapipe 0.10.x and 1.x
import os
import torchvision
import mediapipe as mp
import cv2
import numpy as np

_DETECT_EVERY = int(os.environ.get("MEDIAPIPE_DETECT_EVERY", "4"))

def _get_face_detection():
    # Try old API first
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
        return mp.solutions.face_detection
    # mediapipe 1.x compat - solutions moved to python.solutions
    try:
        from mediapipe.python import solutions as mp_solutions
        return mp_solutions.face_detection
    except ImportError:
        pass
    try:
        import mediapipe.python.solutions.face_detection as fd
        return fd
    except ImportError:
        pass
    raise ImportError("Could not find face_detection in mediapipe %s - try pip install mediapipe==0.10.14" % getattr(mp, "__version__", "unknown"))

class LandmarksDetector:
    def __init__(self):
        self.mp_face_detection = _get_face_detection()
        self.short_range_detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=0)
        self.full_range_detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=1)
        print(f"[LandmarksDetector] MEDIAPIPE_DETECT_EVERY={_DETECT_EVERY} (detect on 1 in every {_DETECT_EVERY} frames)", flush=True)

    def __call__(self, filename):
        video_frames = torchvision.io.read_video(filename, pts_unit='sec')[0].numpy()
        landmarks = self.detect(video_frames, self.short_range_detector)
        if all(element is None for element in landmarks):
            landmarks = self.detect(video_frames, self.short_range_detector)
            assert any(l is not None for l in landmarks), "Cannot detect any frames in the video"
        return landmarks

    def detect(self, video_frames, detector):
        n_frames = len(video_frames)
        landmarks = [None] * n_frames
        keyframe_indices = list(range(0, n_frames, _DETECT_EVERY))
        if (n_frames - 1) not in keyframe_indices:
            keyframe_indices.append(n_frames - 1)
        for idx in keyframe_indices:
            frame = video_frames[idx]
            results = detector.process(frame)
            if not results.detections:
                continue
            landmarks[idx] = self._extract_landmarks(frame, results)
        if _DETECT_EVERY > 1:
            landmarks = self._interpolate(landmarks, n_frames)
        return landmarks

    def _extract_landmarks(self, frame, results):
        ih, iw, _ = frame.shape
        max_id, max_size = 0, 0
        for idx, detected_faces in enumerate(results.detections):
            bboxC = detected_faces.location_data.relative_bounding_box
            bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih))
            bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
            if bbox_size > max_size:
                max_id, max_size = idx, bbox_size
        detected_faces = results.detections[max_id]
        lmx = [
            [int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(i).value].x * iw),
             int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(i).value].y * ih)]
            for i in range(4)
        ]
        return np.array(lmx)

    def _interpolate(self, landmarks, n_frames):
        detected = [i for i, lm in enumerate(landmarks) if lm is not None]
        if not detected:
            return landmarks
        for i in range(0, detected[0]):
            landmarks[i] = landmarks[detected[0]]
        for i in range(detected[-1] + 1, n_frames):
            landmarks[i] = landmarks[detected[-1]]
        for a, b in zip(detected[:-1], detected[1:]):
            if b - a <= 1:
                continue
            lm_a = landmarks[a].astype(float)
            lm_b = landmarks[b].astype(float)
            for i in range(1, b - a):
                t = i / float(b - a)
                landmarks[a + i] = (lm_a + t * (lm_b - lm_a)).astype(int)
        return landmarks

import os

import cv2
import numpy as np
from tqdm import tqdm

from yolo.detector import SpermYOLODetector
from yolo.tracker import Tracker
from yolo.yolo_track import YOLO_INPUT_SIZE


def resize_with_padding(image, target_size):
    h, w, _ = image.shape[:3]
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    padded_image = cv2.copyMakeBorder(resized_image, pad_h, target_size - new_h - pad_h,
                                      pad_w, target_size - new_w - pad_w, cv2.BORDER_CONSTANT,
                                      value=[128, 128, 128])

    return padded_image, scale, pad_w, pad_h


def correct_bbox(bbox, scale, pad_w, pad_h, original_w, original_h):
    x1 = (bbox[0] - pad_w) / scale
    y1 = (bbox[1] - pad_h) / scale
    x2 = (bbox[2] - pad_w) / scale
    y2 = (bbox[3] - pad_h) / scale

    x1 = max(0, min(original_w, x1))
    y1 = max(0, min(original_h, y1))
    x2 = max(0, min(original_w, x2))
    y2 = max(0, min(original_h, y2))

    return [int(x1), int(y1), int(x2), int(y2)]


def calculate_motility_metrics(trajectories, displacement_threshold, straightness_threshold):
    motility_data = {
        "total": 0,
        "motile": 0,
        "progressive": 0,
        "immotile": 0
    }

    for track_id, trajectory in trajectories.items():
        motility_data["total"] += 1

        if len(trajectory) < 2:
            motility_data["immotile"] += 1
            continue

        displacements = np.sqrt(np.sum(np.diff(trajectory, axis=0) ** 2, axis=1))
        total_displacement = np.linalg.norm(np.array(trajectory[-1]) - np.array(trajectory[0]))
        path_length = np.sum(displacements)
        straightness = total_displacement / path_length if path_length > 0 else 0

        if total_displacement > displacement_threshold:
            motility_data["motile"] += 1
            if straightness > straightness_threshold:
                motility_data["progressive"] += 1
        else:
            motility_data["immotile"] += 1

    return motility_data


class YOLOTrackEvaluator:
    def __init__(self, model_path, confidence=0.1):
        self.detector = SpermYOLODetector(model_path=model_path, confidence=confidence)
        self.tracker = Tracker()

    def evaluate_video(self, video_path, video_length=4, displacement_threshold=5, straightness_threshold=0.8):
        """
        Evaluates the sperm motility metrics for a given video.

        :param video_path: Path to the video file.
        :param video_length: How much of the video frames to evaluate in seconds.
        :param displacement_threshold: Threshold for motility displacement.
        :param straightness_threshold: Threshold for progressive motility straightness.
        :return: Dictionary with aggregated motility metrics.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
        frame_limit = fps * video_length  # Total frames to process

        # Aggregated trajectories for motility calculation
        trajectories = {}
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        with tqdm(total=frame_limit, desc=f"Processing video: {video_name}") as pbar:
            for _ in range(frame_limit):
                ret, frame = cap.read()
                if not ret:
                    break

                original_h, original_w = frame.shape[:2]
                resized_frame, scale, pad_w, pad_h = resize_with_padding(frame, YOLO_INPUT_SIZE)

                # Detect objects in the frame
                detections = self.detector.detect(resized_frame)

                # Track detections across frames
                tracking_ids, boxes = self.tracker.track(detections, resized_frame)

                # Update trajectories for motility calculation
                for tracking_id, bbox in zip(tracking_ids, boxes):
                    corrected_bbox = correct_bbox(bbox, scale, pad_w, pad_h, original_w, original_h)
                    center = ((corrected_bbox[0] + corrected_bbox[2]) // 2, (corrected_bbox[1] + corrected_bbox[3]) // 2)

                    if tracking_id not in trajectories:
                        trajectories[tracking_id] = []
                    trajectories[tracking_id].append(center)
                pbar.update(1)

        cap.release()

        # Calculate motility metrics
        motility_data = calculate_motility_metrics(trajectories, displacement_threshold, straightness_threshold)

        return motility_data

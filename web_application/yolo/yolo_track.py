import cv2
import numpy as np
import time
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

from yolo.detector import SpermYOLODetector
from yolo.tracker import Tracker
from yolo.motility import calculate_motility_metrics

MODEL_PATH = "../experiments/model/best_42_Large3e.pt"
# MODEL_PATH = "../runs/detect/train2/weights/best.pt"
VIDEO_PATH = "../../data/shorter_6ms.mp4"

# Set the desired input size for YOLO (640x640)
YOLO_INPUT_SIZE = 640

# Set the desired FPS for frame control
DESIRED_FPS = 3  # Slow down to 5 frames per second


class YOLOTrack:
    def __init__(self, qt_track_container):
        self.detector = SpermYOLODetector(model_path=MODEL_PATH, confidence=0.1)
        self.tracker = Tracker()

        self.qt_track_container = qt_track_container

        self.cap = None

    def yoloTrack(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            print("Error: Unable to open video file.")
            exit()

        # Frame control using a delay (calculated based on desired FPS)
        frame_delay = int(1000 / DESIRED_FPS)

        trajectories = {}
        frames = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            original_h, original_w = frame.shape[:2]

            # Resize the frame with padding to 416x416
            resized_frame, scale, pad_w, pad_h = self.resize_with_padding(frame, YOLO_INPUT_SIZE)

            start_time = time.perf_counter()

            # # Detect objects on the resized 416x416 frame
            detections = self.detector.detect(resized_frame)
            tracking_ids, boxes = self.tracker.track(detections, resized_frame)

            # Draw the bounding boxes on the original frame (resize back)
            for tracking_id, bbox in zip(tracking_ids, boxes):
                corrected_bbox = self.correct_bbox(bbox, scale, pad_w, pad_h, original_w, original_h)

                center = ((corrected_bbox[0] + corrected_bbox[2]) // 2, (corrected_bbox[1] + corrected_bbox[3]) // 2)

                # Update trajectory for the tracking ID
                if tracking_id not in trajectories:
                    trajectories[tracking_id] = []

                trajectories[tracking_id].append(center)

                # Draw the bounding box and tracking ID on the original frame
                cv2.rectangle(frame, (corrected_bbox[0], corrected_bbox[1]), (corrected_bbox[2], corrected_bbox[3]),
                              (0, 0, 255), 2)
                cv2.putText(frame, f"{str(tracking_id)}", (corrected_bbox[0], corrected_bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            frames += 1
            end_time = time.perf_counter()
            fps = 1 / (end_time - start_time)
            print(f"Current fps: {fps}")
            print(f"Current frame: {frames}")

            # Convert frame to QImage and display it in the Qt container
            # Render the frame in the track_container
            self.render_frame_in_track_container(frame)

            # Ensure GUI updates
            # QApplication.processEvents()

            # Frame control: wait for a keypress or based on desired FPS
            key = cv2.waitKey(frame_delay) & 0xFF  # Slows down frame processing

            # Break the loop if 'q' or 'ESC' is pressed
            if key == ord("q") or key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

        # Analyze motility after processing the video
        motility_data = calculate_motility_metrics(trajectories)

        self.write_motility_statistics(motility_data, self.qt_track_container.motility_label)

        print("Motility Statistics:")
        print(f"Total Sperm: {motility_data['total']}")
        print(f"Motile: {motility_data['motile']}")
        print(f"Progressive: {motility_data['progressive']}")
        print(f"Immotile: {motility_data['immotile']}")

        print("frames ", frames)
        return motility_data

    def write_motility_statistics(self, motility_data, motility_label):
        """
        Print the motility statistics on the GUI.
        """
        total_sperm = motility_data["total"]
        motile_sperm = motility_data["motile"]
        progressive_sperm = motility_data["progressive"]
        immotile_sperm = motility_data["immotile"]

        motility_label.setText(
            f"Total Sperm: {total_sperm}\n"
            f"Motile: {motile_sperm}\n"
            f"Progressive: {progressive_sperm}\n"
            f"Immotile: {immotile_sperm}"
        )

    def render_frame_in_track_container(self, frame):
        """
        Render the given frame in the track_container using PyQt6 QGraphicsView and QGraphicsScene.
        """

        # Convert the frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        # Convert the frame to a QImage
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(qt_image)

        # Update the graphics scene
        self.qt_track_container.graphics_scene.clear()
        self.qt_track_container.graphics_scene.addPixmap(pixmap)

        # Ensure the view adjusts to fit the image
        self.qt_track_container.graphics_view.fitInView(
            self.qt_track_container.graphics_scene.itemsBoundingRect(),
            Qt.AspectRatioMode.KeepAspectRatio
        )

    def resize_with_padding(self, image, target_size):
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

    def correct_bbox(self, bbox, scale, pad_w, pad_h, original_w, original_h):
        # Correct bounding box by reversing the scaling and padding
        x1 = (bbox[0] - pad_w) / scale
        y1 = (bbox[1] - pad_h) / scale
        x2 = (bbox[2] - pad_w) / scale
        y2 = (bbox[3] - pad_h) / scale

        # Clip the bounding box to ensure it's within image bounds
        x1 = max(0, min(original_w, x1))
        y1 = max(0, min(original_h, y1))
        x2 = max(0, min(original_w, x2))
        y2 = max(0, min(original_h, y2))

        return [int(x1), int(y1), int(x2), int(y2)]

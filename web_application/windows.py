import numpy as np
import torch
import torch.nn as nn

import sys

from model import SpermCountingCNN
import funct

from yolo.yolo_track import YOLOTrack

from layout_colorwidget import ColorLabel

from PyQt6.QtCore import QSize, Qt, QUrl
from PyQt6.QtGui import QAction
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget,
    QMessageBox, QStackedWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)

from layout_colorwidget import Color


class VideoModelApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Model Runner")
        self.setGeometry(100, 100, 400, 200)

        self.active_layout = "home"

        self.model_path = "../experiments/model/sc_cnn_state_dict.pth"
        self.model = SpermCountingCNN()
        self.load_model(self.model_path)

        self.video_path = "D:\\university\\2024\\BioMed Project\\Sperm-Analysis\\data\\shorter_2s.mp4"
        self.temp_output = "temp_output"

        # Media Player Elements
        self.media_player = QMediaPlayer()
        self.video_widget = QVideoWidget()
        self.audio_output = QAudioOutput()
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.setAudioOutput(self.audio_output)

        # Create a Image Viewer Elements
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)

        # UI Elements - Home
        self.label = QLabel("Upload a video file:", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.prediction_label = QLabel("Prediction will appear here", self)
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.upload_button = QPushButton("Upload Video", self)
        self.upload_button.clicked.connect(self.upload_video)

        self.get_prediction_button = QPushButton("Get Sperm Count Prediction", self)
        self.get_prediction_button.clicked.connect(self.get_prediction)

        self.video_view_button = QPushButton("Go to tracking interface", self)
        self.video_view_button.clicked.connect(self.go_to_video)

        # UI Elements - Video
        self.switch_layout_button = QPushButton("Home", self)
        self.switch_layout_button.clicked.connect(self.switch_layout)

        self.play_video_button = QPushButton("Play uploaded video", self)
        self.play_video_button.clicked.connect(self.play_video)

        self.track_sperm_button = QPushButton("Track sperm in uploaded video", self)
        self.track_sperm_button.clicked.connect(self.track_sperm)

        # UI Elements - Track
        self.motility_label = QLabel("Motility Prediction will appear here", self)
        self.motility_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Layout
        home_layout = self.get_layout(self.active_layout)

        self.home_container = QWidget()
        self.home_container.setLayout(home_layout)

        video_layout = self.get_layout("video")

        self.video_container = QWidget()
        self.video_container.setLayout(video_layout)

        track_layout = self.get_layout("track")

        self.track_container = QWidget()
        self.track_container.setLayout(track_layout)
        self.track_container.setHidden(True)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.home_container)
        self.stack.addWidget(self.video_container)
        self.stack.addWidget(self.track_container)

        self.stack.setCurrentWidget(self.home_container)
        self.setCentralWidget(self.stack)

        self.yolo_track = YOLOTrack(self)

    def get_layout(self, layout_name: str):
        layout = QVBoxLayout()

        match layout_name:
            case "home":
                layout.addWidget(self.label)
                layout.addWidget(self.upload_button)
                layout.addWidget(self.get_prediction_button)
                layout.addWidget(self.prediction_label)
                layout.addWidget(self.video_view_button)
            case "video":
                layout.addWidget(self.switch_layout_button)
                layout.addWidget(self.video_widget)
                layout.addWidget(self.play_video_button)
                layout.addWidget(self.track_sperm_button)
            case "track":
                layout.addWidget(self.switch_layout_button)
                layout.addWidget(self.graphics_view)
                layout.addWidget(self.motility_label)
            case _:
                layout.addWidget(self.switch_layout_button)

        return layout

    def go_to_video(self):
        self.active_layout = "video"
        self.stack.setCurrentWidget(self.video_container)

    def switch_layout(self):
        if self.active_layout == "video":
            self.active_layout = "home"

            self.stack.setCurrentWidget(self.home_container)

    def load_model(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path))

    def upload_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)")

        if file_path:
            self.label.setText(f"Processing: {file_path}")
            self.video_path = file_path
            try:
                # print(self.model)
                self.prediction_label.setText("Prediction will appear here")
                QMessageBox.information(self, "Success", "Video and model loaded successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {e}")
        else:
            self.label.setText("Upload a video file:")

    def play_video(self):
        if self.video_path is None:
            QMessageBox.critical(self, "Error", "Please upload a video first")
        else:
            url = QUrl.fromLocalFile(self.video_path)
            self.media_player.setSource(url)
            self.media_player.play()

    def track_sperm(self):
        # Stop media player if it's playing
        self.media_player.stop()

        if self.video_path is None:
            QMessageBox.critical(self, "Error", "Please upload a video first")
        else:
            self.stack.setCurrentWidget(self.track_container)
            self.yolo_track.yoloTrack(self.video_path)

    def get_prediction(self):
        if self.video_path is None:
            QMessageBox.critical(self, "Error", "Please upload a video first")
        else:
            predicted_sperm_count = funct.get_prediction(self.video_path, self.temp_output, self.model)
            self.prediction_label.setText(
                f"On video {self.video_path} predicted {predicted_sperm_count:.2f} (x10⁶) sperm count")
            QMessageBox.information(self, "Success",
                                    f"Model ran successfully on the video and predicted {predicted_sperm_count:.2f} (x10⁶) sperm count")

import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

import sys

from model import SpermCountingCNN
import funct
from log import log_prediction, LOG_FILE

from yolo.yolo_track import YOLOTrack

from layout_colorwidget import ColorLabel

from PyQt6.QtCore import QSize, Qt, QUrl
from PyQt6.QtGui import QAction
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget,
    QMessageBox, QStackedWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QDialog, QScrollArea, QFrame,
    QDoubleSpinBox
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

        # Sperm volume input element
        self.volume_label = QLabel("Sperm Volume (ml):", self)
        self.volume_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.volume_input = QDoubleSpinBox(self)
        self.volume_input.setDecimals(2)
        self.volume_input.setRange(0.0, 100.0)
        self.volume_input.setValue(0.0)
        self.volume_input.setSuffix(" ml")

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

        # Logs
        self.view_logs_button = QPushButton("View Prediction Logs", self)
        self.view_logs_button.clicked.connect(self.view_logs)
        home_layout.addWidget(self.view_logs_button)
        home_layout.addWidget(self.volume_label)
        home_layout.addWidget(self.volume_input)

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
        # Show confirmation dialog
        proceed = self.show_confirmation_dialog(
            "Confirmation",
            "You will see values without a doctor's interpretation. Are you sure you want to proceed?"
        )
        if not proceed:
            return  # Cancel the action if the user selects "No"

        # Stop media player if it's playing
        self.media_player.stop()

        if self.video_path is None:
            QMessageBox.critical(self, "Error", "Please upload a video first")
        else:

            self.stack.setCurrentWidget(self.track_container)
            motility_data = self.yolo_track.yoloTrack(self.video_path)
            sperm_volume = self.volume_input.value()

            # Log the motility prediction
            log_prediction(self.video_path, "motility", motility_data, sperm_volume)

    def get_prediction(self):
        # Show confirmation dialog
        proceed = self.show_confirmation_dialog(
            "Confirmation",
            "You will see values without a doctor's interpretation. Are you sure you want to proceed?"
        )
        if not proceed:
            return  # Cancel the action if the user selects "No"

        if self.video_path is None or not os.path.isfile(self.video_path):
            QMessageBox.critical(self, "Error", "Please upload a video first")
        else:
            predicted_sperm_count = funct.get_prediction(self.video_path, self.temp_output, self.model)
            sperm_volume = self.volume_input.value()
            self.prediction_label.setText(
                f"On video {self.video_path} predicted {predicted_sperm_count:.2f} (x10⁶) sperm count")
            QMessageBox.information(self, "Success",
                                    f"Model ran successfully on the video and predicted {predicted_sperm_count:.2f} (x10⁶) sperm count")

            # Log the prediction
            log_prediction(self.video_path, "sperm_count", predicted_sperm_count, sperm_volume)

    def show_confirmation_dialog(self, title, message):
        confirmation = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        return confirmation == QMessageBox.StandardButton.Yes

    def view_logs(self):
        try:
            with open(LOG_FILE, "r") as file:
                logs = json.load(file)
        except FileNotFoundError:
            logs = []
        except json.JSONDecodeError:
            QMessageBox.critical(self, "Error", "Log file is corrupted or contains invalid data.")
            return

        if not logs:
            QMessageBox.information(self, "Prediction Logs", "No logs available.")
            return

        # Create the scrollable dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Prediction Logs")
        dialog.resize(600, 400)

        layout = QVBoxLayout(dialog)

        scroll_area = QScrollArea(dialog)
        scroll_area.setWidgetResizable(True)

        container_widget = QWidget()
        container_layout = QVBoxLayout(container_widget)

        # Format and style each log entry
        for log in reversed(logs):
            log_frame = QFrame()
            log_frame.setFrameShape(QFrame.Shape.Box)
            log_frame.setFrameShadow(QFrame.Shadow.Raised)
            log_frame.setStyleSheet("background-color: #f9f9f9; border: 1px solid #ccc; margin: 5px; padding: 10px;")

            log_layout = QVBoxLayout(log_frame)

            # Format date
            formatted_date = datetime.fromisoformat(log["timestamp"]).strftime("%d %B %Y, %H:%M:%S")
            date_label = QLabel(f"<b>Date:</b> {formatted_date}")
            date_label.setStyleSheet("color: #555; font-size: 12px;")

            # Display video path
            video_label = QLabel(f"<b>Video:</b> {log['video_path']}")
            video_label.setStyleSheet("color: #333; font-size: 14px;")

            volume_label = QLabel(f"<b>Sperm Volume:</b> {log['sperm_volume']} ml")
            volume_label.setStyleSheet("color: #333; font-size: 14px;")

            # Display prediction type and value
            if log["prediction_type"] == "motility":
                motility = log["prediction_value"]
                prediction_label = QLabel(
                    f"<b>Prediction:</b> Motility<br>"
                    f"  Total: {motility['total']}<br>"
                    f"  Motile: {motility['motile']}<br>"
                    f"  Progressive: {motility['progressive']}<br>"
                    f"  Immotile: {motility['immotile']}"
                )
            else:
                prediction_label = QLabel(f"<b>Prediction:</b> Sperm Count: {log['prediction_value']:.2f} (x10⁶)")
            prediction_label.setStyleSheet("color: #444; font-size: 14px;")

            # Add widgets to the log layout
            log_layout.addWidget(date_label)
            log_layout.addWidget(video_label)
            log_layout.addWidget(volume_label)
            log_layout.addWidget(prediction_label)

            # Add the styled log frame to the container
            container_layout.addWidget(log_frame)

        # Add the container to the scroll area
        scroll_area.setWidget(container_widget)
        layout.addWidget(scroll_area)

        dialog.exec()

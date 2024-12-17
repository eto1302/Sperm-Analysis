import numpy as np
import torch 
import torch.nn as nn

import sys

from model import SpermCountingCNN
import funct

from layout_colorwidget import ColorLabel

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QAction

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget,
    QMessageBox
)

from layout_colorwidget import Color



class VideoModelApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Model Runner")
        self.setGeometry(100, 100, 400, 200)

        self.model_path = "../experiments/model/sc_cnn_state_dict.pth"
        self.model = SpermCountingCNN() 
        self.load_model(self.model_path)

        self.video_path = None
        self.temp_output = "temp_output"

        # UI Elements
        self.label = QLabel("Upload a video file:", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.prediction_label = QLabel("Prediction will appear here", self)
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.upload_button = QPushButton("Upload Video", self)
        self.upload_button.clicked.connect(self.upload_video)

        self.get_prediction_button = QPushButton("Get Sperm Count Prediction", self)
        self.get_prediction_button.clicked.connect(self.get_prediction)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.get_prediction_button)
        layout.addWidget(self.prediction_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

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


    def get_prediction(self):        
        if self.video_path is None: 
            QMessageBox.critical(self, "Error", "Please upload a video first")
        else:
            predicted_sperm_count = funct.get_prediction(self.video_path, self.temp_output, self.model)
            self.prediction_label.setText(f"On video {self.video_path} predicted {predicted_sperm_count:.2f} (x10⁶) sperm count")
            QMessageBox.information(self, "Success", f"Model ran successfully on the video and predicted {predicted_sperm_count:.2f} (x10⁶) sperm count")

        

import numpy as np
import torch 
import torch.nn as nn
import sys

from model import SpermCountingCNN

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

        # UI Elements
        self.label = QLabel("Upload a video file:", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.upload_button = QPushButton("Upload Video", self)
        self.upload_button.clicked.connect(self.upload_video)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.upload_button)

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
            try:
                print(self.model)
                QMessageBox.information(self, "Success", "Model ran successfully on the video!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {e}")
        else:
            self.label.setText("Upload a video file:")



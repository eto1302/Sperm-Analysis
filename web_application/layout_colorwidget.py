from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QWidget, QLabel


class ColorLabel(QLabel): 
    def __init__(self, text, color):
        super().__init__(text=text)
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)


class Color(QWidget):
    def __init__(self, color):
        super().__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget, QPushButton, QSlider, QLabel, QHBoxLayout, QVBoxLayout, QDialog

import torch

class CanvasPanel(QWidget):

    def __init__(self,canvas_size):
        super().__init__()

        self.canvas_size = canvas_size
        self.brush_size = 10
        self.brush_color = Qt.GlobalColor.black

        self.drawing = False
        self.last_pos = QPoint()

        self.drawing_surface = QImage(self.canvas_size[0], self.canvas_size[1] ,QImage.Format.Format_RGB32)
        self.drawing_surface.fill(Qt.GlobalColor.white)

        self.clear_btn = QPushButton("Clear")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(10)
        self.slider.setMaximum(15)
        self.slider.setValue(self.brush_size)

        self.slider_label = QLabel(str(self.brush_size))

        # Layout

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(0,0,0,0)
        self.root.setSpacing(10)

        controls_widget = QWidget()
        controls = QHBoxLayout(controls_widget)
        controls.addWidget(self.clear_btn)
        controls.addWidget(QLabel("Pen Size"))
        controls.addWidget(self.slider)
        controls.addWidget(self.slider_label)

        self.root.addLayout(controls)

        controls_widget = QWidget()
        controls = QHBoxLayout(controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(10)

        controls.addWidget(self.clear_btn)
        controls.addWidget(QLabel("Pen Size"))
        controls.addWidget(self.slider)
        controls.addWidget(self.slider_label)

        self.root.addWidget(controls_widget)
        self.controls_widget = controls_widget

        controls_height = self.root.itemAt(0).sizeHint().height()

        self.setMinimumSize(self.canvas_size[0], self.canvas_size[1] + controls_height)

        self.clear_btn.clicked.connect(self.clear)
        self.slider.valueChanged.connect(self.set_pensize)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            height_offset = self.canvas_offset_y()
            pos = event.position().toPoint()

            if pos.y() < height_offset:
                return

            self.drawing = True
            self.last_pos = QPoint(pos.x(), pos.y() - height_offset)


    def mouseMoveEvent(self, event):
        if not self.drawing:
            return

        height_offset = self.canvas_offset_y()
        pos = event.position().toPoint()

        if pos.y() < height_offset:
            return

        current = QPoint(pos.x(), pos.y() - height_offset)

        painter = QPainter(self.drawing_surface)
        pen = QPen(self.brush_color)
        pen.setWidth(self.brush_size)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        painter.drawLine(self.last_pos, current)
        painter.end()

        self.last_pos = current
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        y0 = self.canvas_offset_y()
        x0 = (self.width() - self.drawing_surface.width()) // 2
        painter.drawImage(x0, y0, self.drawing_surface)

    def clear(self):
        self.drawing_surface.fill(Qt.GlobalColor.white)
        self.update()

    def set_pensize(self, value:int):
        self.brush_size = int(value)
        self.slider_label.setText(str(self.brush_size))

    def to_tensor(self):

        gray_scale_img = self.drawing_surface.convertToFormat(QImage.Format.Format_Grayscale8)

        resized_img = gray_scale_img.scaled(
            32,32,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        pointer_to_img_in_mem = resized_img.bits()
        pointer_to_img_in_mem.setsize(32*32)

        img_tensor = torch.frombuffer(pointer_to_img_in_mem, dtype=torch.uint8).reshape(32,32)
        img_tensor = (255 - img_tensor).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

        return img_tensor

    # def canvas_offset_y(self):
    #     return self.controls_widget.height() + self.root.spacing()





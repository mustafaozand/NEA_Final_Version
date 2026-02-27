# from __future__ import annotations

import math
import random

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union


from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QPainter, QPen, QBrush, QGuiApplication, QAction, QPixmap, QColor
from PyQt6.QtWidgets import QFrame, QToolTip, QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QApplication

import sys
import torch


@dataclass
class NetworkSnapshot:
    layer_sizes: List[int]
    activation: Optional[List[List[float]]] = None
    biases: Optional[List[List[float]]] = None
    weights: Optional[List[List[List[float]]]] = None

@dataclass(frozen=True)
class DisplayNode:
    kind: str
    pos: QPointF
    idx: Optional[int] = None


class NetworkViewApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Network View")
        self.setGeometry(0, 0, 700, 500)
        # screen = QGuiApplication.primaryScreen()
        self.network_vis = NetworkView()
        central = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.network_vis)
        self.setCentralWidget(central)
        self.resize(900, 650)





class NetworkView(QFrame):
    def __init__(self):
        super().__init__()

        self.setObjectName("NetworkView")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True) # Enables the hovering feature

        self.margin = 24
        self.node_rad = 12
        self.node_gap = 60
        self.edge_hit_threshold = 6.0 # edge_hit_threshold is how far your mouse has to be in order to activate the mini window with info about an edge

        self.snapshot = None # holds the latest values

        self.gray_scale_values = []

        self.node_hitboxes = {}
        self.edge_segments = {}

        self.last_hover_key = None # this will prevent tooltip spam, because qt would show the tooltip every tiny mouse movement

        self.set_snapshot(self.make_dummy_snapshot([784, 16,16,10]))

        self.compute_grayscale_values()


    def set_snapshot(self, snap):
        self.snapshot = snap
        self.update()

    def make_dummy_snapshot(self, layer_sizes):

        activations = []
        biases = []
        weights = []

        for n in layer_sizes:
            activations.append([random.random() for  _ in range(n)])

        for li in range(1, len(layer_sizes)):
            n_from = layer_sizes[li - 1]
            n_to = layer_sizes[li]

            biases.append([random.uniform(-1,1) for _ in range(n_to)])

            w_layer = []

            for _to in range(n_to):
                w_layer.append([random.uniform(-1, 1) for _ in range(n_from)])

            weights.append(w_layer)

        return NetworkSnapshot(layer_sizes, activations, biases, weights)


    def paintEvent(self,event):
        super().paintEvent(event)

        painter = QPainter(self)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        self.node_hitboxes.clear()
        self.edge_segments.clear()

        if self.snapshot is None or not self.snapshot.layer_sizes:
            painter.drawtext = (self.rect(), Qt.AlignmentFlag.AlignCenter, "Network View (no data yet")

        layers = self.compute_display_layout(self.snapshot.layer_sizes)

        self.draw_edges(painter, layers)

        self.draw_nodes(painter, layers)

    def select_indices(self, n, head: int = 3, tail: int = 2):

        if n <= (head + tail + 1):
            return list(range(n))

        out = list(range(head))
        out.append(None)
        out.extend(range(n - tail, n))

        return out

    def compute_display_layout(self, layer_sizes: Sequence[int]) -> List[List[DisplayNode]]:
        """
        Computes the on-screen positions for each layer's display items (nodes + ellipsis).
        Returns:
          layers[layer_index] = list[DisplayNode]
        """
        # Widget width and height in pixels.
        w = self.width()
        h = self.height()

        # Compute drawing bounds with margins.
        left = self.margin
        top = self.margin
        right = w - self.margin
        bottom = h - self.margin

        # Number of layers (columns).
        L = len(layer_sizes)

        # Horizontal space available for layer columns.
        usable_w = max(1.0, right - left)

        # Compute x positions evenly across usable width.
        # If L==1, max(1, L-1) avoids division by 0.
        xs = [left + (usable_w * i) / max(1, L - 1) for i in range(L)]

        layers: List[List[DisplayNode]] = []

        for layer_i, n_neurons in enumerate(layer_sizes):
            chosen = self.select_indices(n_neurons, head=3, tail=2)
            visible = len(chosen)

            # Determine y positions for this layer's visible items,
            # centered vertically within the panel.
            if visible == 1:
                ys = [(top + bottom) / 2.0]
            else:
                total_h = (visible - 1) * self.node_gap
                start_y = (top + bottom) / 2.0 - total_h / 2.0
                ys = [start_y + i * self.node_gap for i in range(visible)]

            col: List[DisplayNode] = []
            for y, idx in zip(ys, chosen):
                pos = QPointF(xs[layer_i], y)
                if idx is None:
                    col.append(DisplayNode(kind="ellipsis", pos=pos, idx=None))
                else:
                    col.append(DisplayNode(kind="node", pos=pos, idx=int(idx)))

            layers.append(col)

        return layers

    def draw_edges(self, painter: QPainter, layers: List[List[DisplayNode]]) -> None:
        """
        Draw lines between visible nodes in adjacent layers.
        Also store edge line segments for hover detection.
        """
        # Create a pen for edges (light gray lines).
        pen = QPen(Qt.GlobalColor.lightGray)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # For each pair of adjacent layers
        for layer_from in range(len(layers) - 1):
            # Only real nodes participate in edges; ellipsis does not.
            left_nodes = [dn for dn in layers[layer_from] if dn.kind == "node"]
            right_nodes = [dn for dn in layers[layer_from + 1] if dn.kind == "node"]

            # Draw fully-connected edges between displayed nodes.
            for a in left_nodes:
                for b in right_nodes:
                    painter.drawLine(a.pos, b.pos)

                    # Save the segment for hover hit-testing.
                    # Key includes (layer_from, from_idx, to_idx)
                    self.edge_segments[(layer_from, a.idx, b.idx)] = (a.pos, b.pos)

    def draw_nodes(self, painter: QPainter, layers: List[List[DisplayNode]]) -> None:
        """
        Draw neuron circles and ellipsis markers, and store node hitboxes.
        """
        r = self.node_rad

        node_pen = QPen(Qt.GlobalColor.black)
        node_pen.setWidth(1)

        node_brush = QBrush(Qt.GlobalColor.white)

        ellipsis_pen = QPen(Qt.GlobalColor.darkGray)
        ellipsis_pen.setWidth(1)

        gray_scale = self.compute_grayscale_values()

        for layer_i, col in enumerate(layers):
            for dn in col:
                if dn.kind == "node":
                    painter.setPen(node_pen)

                    value = gray_scale[layer_i][dn.idx]  # 0..255 float
                    value = max(0, min(255, int(value)))  # clamp + convert to int
                    color = QColor(value, value, value)  # grayscale QColor

                    node_brush = QBrush(color)
                    painter.setBrush(node_brush)

                    # Create the bounding rectangle of the circle.
                    rect = QRectF(dn.pos.x() - r, dn.pos.y() - r, 2 * r, 2 * r)

                    # Draw the circle.
                    painter.drawEllipse(rect)

                    # Save hitbox for hover detection.
                    self.node_hitboxes[(layer_i, dn.idx)] = rect

                else:
                    # Ellipsis marker: draw "⋯" centered at dn.pos
                    painter.setPen(ellipsis_pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)

                    text_rect = QRectF(dn.pos.x() - 20, dn.pos.y() - 12, 40, 24)
                    painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, "⋯")

    def mouseMoveEvent(self, event):
        """
        Called when mouse moves over the widget (because mouse tracking is on).
        We compute what the mouse is hovering and show a tooltip accordingly.
        """
        if self.snapshot is None:
            return

        pos = event.position()  # QPointF in widget coordinates

        hit = self.hit_test(pos)
        if hit is None:
            # If we were previously hovering something, hide tooltip now.
            if self.last_hover_key is not None:
                QToolTip.hideText()
                self.last_hover_key = None
            return

        kind, key_tuple, text = hit
        hover_key = (kind, key_tuple)

        # Only update tooltip if hover changed.
        if hover_key != self.last_hover_key:
            QToolTip.showText(self.mapToGlobal(pos.toPoint()), text, self)
            self.last_hover_key = hover_key

    def leaveEvent(self, event):
        """
        Called when mouse leaves the widget.
        We hide any tooltip and reset hover state.
        """
        QToolTip.hideText()
        self.last_hover_key = None
        super().leaveEvent(event)

    def hit_test(self, pos: QPointF) -> Optional[Tuple[str, Tuple[int, ...], str]]:
        """
        Returns:
          ("node", (layer_i, neuron_i), tooltip_text)
        or:
          ("edge", (layer_from, from_i, to_i), tooltip_text)
        or None if nothing is under the mouse.
        """

        # 1) Check nodes first (more intuitive)
        for (layer_i, neuron_i), rect in self.node_hitboxes.items():
            if rect.contains(pos):
                return ("node", (layer_i, neuron_i), self.node_tooltip(layer_i, neuron_i))

        # 2) Check edges by distance-to-segment
        best: Optional[Tuple[float, Tuple[int, int, int]]] = None

        for (layer_from, from_i, to_i), (a, b) in self.edge_segments.items():
            d = self.dist_point_to_segment(pos, a, b)

            if d <= self.edge_hit_threshold:
                # Pick the closest edge if multiple are near.
                if best is None or d < best[0]:
                    best = (d, (layer_from, from_i, to_i))

        if best is not None:
            layer_from, from_i, to_i = best[1]
            return ("edge", (layer_from, from_i, to_i), self.edge_tool_tip(layer_from, from_i, to_i))

        return None

    def node_tooltip(self, layer_i: int, neuron_i: int) -> str:
        """
        Builds tooltip text for a node.
        Node should show activation and bias.
        """
        a = None
        b = None

        if self.snapshot.activation is not None:
            a = self.snapshot.activation[layer_i][neuron_i]

        # biases are stored per non-input layer:
        # biases[0] corresponds to layer 1,
        # so for layer_i>0 we read biases[layer_i-1].
        if layer_i > 0 and self.snapshot.biases is not None:
            b = self.snapshot.biases[layer_i - 1][neuron_i]

        nodes = [f"Layer {layer_i}  Neuron {neuron_i}"]
        if a is not None:
            nodes.append(f"activation a = {a:.4f}")
        if b is not None:
            nodes.append(f"bias b = {b:.4f}")
        return "\n".join(nodes)

    def edge_tool_tip(self, layer_from: int, from_i: int, to_i: int) -> str:
        """
        Builds tooltip text for an edge.
        Edge should show weight.
        """
        w = None

        # weights[layer_from] maps layer_from -> layer_from+1, shape [to][from]
        if self.snapshot.weights is not None:
            w = self.snapshot.weights[layer_from][to_i][from_i]

        lines = [f"Weight (L{layer_from}:{from_i} → L{layer_from + 1}:{to_i})"]
        if w is not None:
            lines.append(f"w = {w:.4f}")
        return "\n".join(lines)

    @staticmethod
    def dist_point_to_segment(p: QPointF, a: QPointF, b: QPointF) -> float:
        """
        Computes shortest distance from point p to the line segment a-b.

        This is needed because edges are not rectangles, so we can't just 'contains()' them.
        """
        ax, ay = a.x(), a.y()
        bx, by = b.x(), b.y()
        px, py = p.x(), p.y()

        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay

        ab_len2 = abx * abx + aby * aby

        # If segment is extremely small, treat it as a point.
        if ab_len2 <= 1e-9:
            return math.hypot(px - ax, py - ay)

        # Project AP onto AB to get parameter t along the segment:
        # t=0 means closest at A, t=1 means closest at B.
        t = (apx * abx + apy * aby) / ab_len2
        t = max(0.0, min(1.0, t))  # clamp to segment

        # Closest point C on the segment:
        cx = ax + t * abx
        cy = ay + t * aby

        return math.hypot(px - cx, py - cy)

    def compute_grayscale_values(self):
        activation = self.snapshot.activation
        print(f"Activation values: {activation}")

        # (curr_activation / max(activation)) * 255

        for layer_i, layer_act in enumerate(activation):
            max_a = max(layer_act)

            layer_gray = [(a / max_a) * 255 for a in layer_act]

            self.gray_scale_values.append(layer_gray)

        return self.gray_scale_values


        print(f"Grayscale values: {self.gray_scale_values}")




def main():
    app = QApplication(sys.argv)
    window = NetworkViewApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()








"""SAHI tiled-inference parameters dialog (additive, standalone).

Standalone QDialog so the auto-labeling .ui file stays untouched.
Shown only for SAHI models via a programmatic toolbar button.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QVBoxLayout,
)

from anylabeling.services.sahi_params import (
    SAHI_DEFAULT_OVERLAP_RATIO,
    SAHI_DEFAULT_SLICE_HEIGHT,
    SAHI_DEFAULT_SLICE_WIDTH,
    SAHI_MAX_OVERLAP,
    SAHI_MAX_SLICE,
    SAHI_MIN_OVERLAP,
    SAHI_MIN_SLICE,
    clamp_sahi_params,
)


class SahiParamsDialog(QDialog):
    """Edit SAHI slice height/width and overlap ratio."""

    def __init__(
        self,
        parent=None,
        slice_height: int = SAHI_DEFAULT_SLICE_HEIGHT,
        slice_width: int = SAHI_DEFAULT_SLICE_WIDTH,
        overlap_ratio: float = SAHI_DEFAULT_OVERLAP_RATIO,
    ):
        super().__init__(parent)
        slice_h, slice_w, overlap = clamp_sahi_params(
            slice_height, slice_width, overlap_ratio
        )
        self._slice_h = slice_h
        self._slice_w = slice_w
        self._overlap = overlap
        self.slice_spin_h: Optional[QSpinBox] = None
        self.slice_spin_w: Optional[QSpinBox] = None
        self.overlap_spin: Optional[QDoubleSpinBox] = None
        self.init_ui()

    def init_ui(self) -> None:
        self.setWindowTitle(self.tr("SAHI Tiled Inference"))
        self.setMinimumWidth(320)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.slice_spin_h = QSpinBox(self)
        self.slice_spin_h.setRange(SAHI_MIN_SLICE, SAHI_MAX_SLICE)
        self.slice_spin_h.setSingleStep(64)
        self.slice_spin_h.setValue(self._slice_h)
        form.addRow(self.tr("Slice height"), self.slice_spin_h)
        self.slice_spin_w = QSpinBox(self)
        self.slice_spin_w.setRange(SAHI_MIN_SLICE, SAHI_MAX_SLICE)
        self.slice_spin_w.setSingleStep(64)
        self.slice_spin_w.setValue(self._slice_w)
        form.addRow(self.tr("Slice width"), self.slice_spin_w)
        self.overlap_spin = QDoubleSpinBox(self)
        self.overlap_spin.setRange(SAHI_MIN_OVERLAP, SAHI_MAX_OVERLAP)
        self.overlap_spin.setSingleStep(0.05)
        self.overlap_spin.setDecimals(2)
        self.overlap_spin.setValue(self._overlap)
        form.addRow(self.tr("Overlap ratio"), self.overlap_spin)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_params(self) -> Optional[Tuple[int, int, float]]:
        """Show dialog; return clamped (h, w, overlap) or None."""
        if self.exec() == QDialog.DialogCode.Accepted:
            return clamp_sahi_params(
                self.slice_spin_h.value(),
                self.slice_spin_w.value(),
                self.overlap_spin.value(),
            )
        return None

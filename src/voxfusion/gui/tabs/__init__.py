"""GUI tab widget builders for VoxFusion.

Each tab class is responsible for constructing the widgets within its
tab frame.  All application state and methods remain on ``TranscriptionGUI``;
the tab objects hold a back-reference (``self._gui``) and delegate to it so
that no other code needs to change.

This is **Phase 1** of ARCH-3: widget-construction extraction.
Phase 2 (moving per-tab methods and state vars into the tab classes) can
proceed incrementally once the boundary is established.
"""

from voxfusion.gui.tabs.file_tab import FileTranscriptionTab
from voxfusion.gui.tabs.live_tab import LiveCaptureTab

__all__ = ["FileTranscriptionTab", "LiveCaptureTab"]

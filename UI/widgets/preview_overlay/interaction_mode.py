from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from UI.widgets.camera_preview import CameraPreview


@dataclass(frozen=True)
class PreviewModeSpec:
    """
    Declarative description of what a tab or wizard step wants the shared
    camera preview to do while it's the one showing it — replaces the
    three ad hoc mechanisms tabs previously used (Navigate relying on
    defaults, Measurement's single fan-out boolean, each calibration
    wizard's hand-rolled overlay save/restore).

    ``visible_overlays`` is ``None`` by default, meaning "don't manage
    these overlays at all" — most modes (Navigate, Measurement) have
    never touched crosshair/grid/focus/etc., and pushing a mode shouldn't
    start clobbering toggles the user set via the toolbar buttons. Pass a
    concrete ``frozenset`` only when the mode genuinely owns a subset of
    these overlays for its own purposes (the calibration wizards) — the
    exact prior value of every managed overlay is restored on pop,
    regardless of what it was set to while this mode was active.
    """

    name: str
    click_to_move: bool = True
    scroll_zooms: bool = False
    measurement_placement: bool = False
    visible_overlays: frozenset[str] | None = None


NAVIGATE_MODE = PreviewModeSpec(name="navigate")

MEASUREMENT_MODE = PreviewModeSpec(
    name="measurement",
    click_to_move=False,
    scroll_zooms=True,
    measurement_placement=True,
)

CALIBRATION_LINE_MODE = PreviewModeSpec(
    name="calibration-line",
    click_to_move=False,
    scroll_zooms=True,
    measurement_placement=True,
)
"""
Placing a manual DPI-calibration reference line — the same underlying
placement gesture as MEASUREMENT_MODE (see MeasurementOverlay's
CALIBRATION_KIND), but pushed independently so any tab or wizard step
can offer manual calibration, not just the Measurement tab.
"""


@dataclass(frozen=True)
class _Snapshot:
    """Concrete preview state captured immediately before a push, so pop can restore exactly that — not whatever some other mode declares."""

    click_to_move_suppressed: bool
    measurement_placement_active: bool
    scroll_zooms: bool
    overlays: dict[str, bool] | None


class ModeToken:
    """
    Returned by ``PreviewModeController.push()``. Calling ``pop()``
    restores the exact preview state captured just before this mode was
    pushed — safe to call more than once. Tokens must be popped in the
    reverse order they were pushed (a plain stack); this holds for every
    current caller (each tab/wizard step owns at most one token at a
    time).
    """

    def __init__(self, controller: "PreviewModeController") -> None:
        self._controller = controller
        self._popped = False

    def pop(self) -> None:
        if self._popped:
            return
        self._popped = True
        self._controller._pop(self)


class PreviewModeController:
    """
    Owned by CameraPreview. Applies PreviewModeSpec state as a stack of
    snapshot/restore pairs, so a mode pushed while another is active (or
    while the user has independently toggled a toolbar overlay button)
    restores correctly on pop — no hand-tracked "state before" fields
    required in the tabs/wizards that use this.
    """

    _MANAGED_OVERLAYS = ("crosshair", "grid", "focus", "inspect_calibration", "red_mark", "background")

    def __init__(self, preview: "CameraPreview") -> None:
        self._preview = preview
        self._stack: list[tuple[ModeToken, PreviewModeSpec, _Snapshot]] = []

    @property
    def current(self) -> PreviewModeSpec:
        return self._stack[-1][1] if self._stack else NAVIGATE_MODE

    def push(self, spec: PreviewModeSpec) -> ModeToken:
        token = ModeToken(self)
        snapshot = self._capture(spec)
        self._stack.append((token, spec, snapshot))
        self._apply(spec)
        return token

    @contextmanager
    def scoped(self, spec: PreviewModeSpec) -> Iterator[None]:
        token = self.push(spec)
        try:
            yield
        finally:
            token.pop()

    def _pop(self, token: ModeToken) -> None:
        """
        Callers are expected to pop in LIFO order (see ModeToken's
        docstring), but if one doesn't, only restore *token*'s own
        snapshot when it was actually the top of the stack — otherwise
        the current top's applied state is still authoritative and must
        be left alone; the out-of-order token is just spliced out so its
        snapshot doesn't leak, and it's re-applied whenever the real top
        eventually pops in turn.
        """
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] is token:
                is_top = i == len(self._stack) - 1
                _, _, snapshot = self._stack.pop(i)
                if is_top:
                    self._restore(snapshot)
                return

    def _capture(self, spec: PreviewModeSpec) -> _Snapshot:
        preview = self._preview
        label = preview._video_label
        overlays = None
        if spec.visible_overlays is not None:
            overlays = {name: getattr(preview.overlays, name) for name in self._MANAGED_OVERLAYS}
        return _Snapshot(
            click_to_move_suppressed=label.click_to_move_suppressed,
            measurement_placement_active=label.measurement_mode_active,
            scroll_zooms=preview._scroll_zooms_mode,
            overlays=overlays,
        )

    def _apply(self, spec: PreviewModeSpec) -> None:
        preview = self._preview
        preview._video_label.set_click_to_move_suppressed(not spec.click_to_move)
        preview._video_label.set_measurement_mode_active(spec.measurement_placement)
        preview._measurement_overlay.set_enabled(spec.measurement_placement)
        preview._scroll_zooms_mode = spec.scroll_zooms
        if spec.visible_overlays is not None:
            overlays = preview.overlays
            for name in self._MANAGED_OVERLAYS:
                setattr(overlays, name, name in spec.visible_overlays)
        preview._video_label.update()

    def _restore(self, snapshot: _Snapshot) -> None:
        preview = self._preview
        preview._video_label.set_click_to_move_suppressed(snapshot.click_to_move_suppressed)
        preview._video_label.set_measurement_mode_active(snapshot.measurement_placement_active)
        preview._measurement_overlay.set_enabled(snapshot.measurement_placement_active)
        preview._scroll_zooms_mode = snapshot.scroll_zooms
        if snapshot.overlays is not None:
            overlays = preview.overlays
            for name, value in snapshot.overlays.items():
                setattr(overlays, name, value)
        preview._video_label.update()

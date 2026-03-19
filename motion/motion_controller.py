from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Callable

import serial
import serial.tools.list_ports

from common.logger import info, error, warning, debug

from .models import Position
from .motion_config import MotionSystemSettings, MotionSystemSettingsManager

_NM_PER_MM = 1_000_000


def _probe_port(
    port_device: str,
    baud: int,
    indicators: list[str],
    request: bytes = b"M115\r\n",
    read_window_s: float = 10,
    min_lines: int = 3,
) -> tuple[serial.Serial | None, list[str]]:
    """
    Try to identify a Marlin-like printer on a single serial port.

    Returns (serial_connection, response_lines) on success, or
    (None, response_lines) on failure.  On success the connection is left
    open for the caller.
    """
    responses: list[str] = []
    ser: serial.Serial | None = None
    success = False
    try:
        ser = serial.Serial(port_device, baudrate=baud, timeout=0.25, write_timeout=1)

        # Some controllers reset on open due to DTR; allow them to chatter.
        start = time.time()
        quiet_since = start
        while time.time() - start < 2.0:
            while ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    responses.append(line)
                    quiet_since = time.time()
            if time.time() - quiet_since > 0.25:
                break
            time.sleep(0.05)

        ser.reset_input_buffer()
        ser.write(request)

        start = time.time()
        while time.time() - start < read_window_s:
            if ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    responses.append(line)
                    if any(ind in line for ind in indicators):
                        success = True
                        break
                    if len(responses) >= min_lines and time.time() - start > 3:
                        success = True
                        break
            else:
                time.sleep(0.05)

        return (ser, responses) if success else (None, responses)

    except Exception:
        return None, responses

    finally:
        if ser is not None and ser.is_open and not success:
            try:
                ser.close()
            except Exception:
                pass


class MotionState:
    """Current lifecycle state of the motion controller."""
    CONNECTING = "connecting"    # Worker thread is still starting up
    READY      = "ready"         # Connected, homed, and accepting commands
    FAULTED    = "faulted"       # Runtime fault (bad G-code response, timeout, etc.)
    FAILED     = "failed"        # Could not connect at all during initialisation


class MotionFault(Exception):
    """Raised when the printer reports an unrecoverable error."""


class MotionTimeout(Exception):
    """Raised when the printer does not respond within the expected window."""


@dataclass
class _Command:
    gcode: str
    message: str | None = None
    log: bool = False


class MotionController:
    """
    Minimal motion controller for a Marlin-based 3D printer / motion system.

    Responsibilities
    ----------------
    - Discover the printer port via _probe_port.
    - Home on startup.
    - Accept G-code move commands via a thread-safe queue.
    - Track the current position in nanometres.
    - Provide move_axis and move_to_position helpers.
    - Provide set_speed to change the jog step size.
    - Expose a simple message-listener hook for UI feedback.
    """

    def __init__(self) -> None:
        self._config_manager = MotionSystemSettingsManager()
        self.config: MotionSystemSettings = self._config_manager.load()

        self.position = Position(0, 0, 0)
        self.speed: int = self.config.step_size  # nanometres per jog step

        self.faulted = False

        self._ready = threading.Event()
        self._init_error: Exception | None = None

        self._stop_event = threading.Event()

        self._command_queue: Queue[_Command] = Queue()
        self._message_listeners: list[Callable[[str, bool], None]] = []

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> str:
        """Return the current :class:`MotionState` of the controller."""
        if not self._ready.is_set():
            return MotionState.CONNECTING
        if self._init_error is not None:
            return MotionState.FAILED
        if self.faulted:
            return MotionState.FAULTED
        return MotionState.READY

    def is_ready(self) -> bool:
        """Return True once the printer has been found and homed."""
        return self.get_state() == MotionState.READY

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """
        Block until the controller is ready (or *timeout* seconds elapse).

        Returns True if ready, False if timed out.
        Raises the initialisation exception if connection failed.
        """
        ready = self._ready.wait(timeout)
        if self._init_error is not None:
            raise self._init_error
        return ready

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        try:
            self._connect()
        except Exception as exc:
            self._init_error = exc
            self._ready.set()
            return

        self._home()
        self._ready.set()
        self._run_loop()

    def _connect(self) -> None:
        """Probe available serial ports and open the first Marlin printer found."""
        baud = self.config.baud_rate
        indicators = [self.config.FIRMWARE_NAME, self.config.MACHINE_TYPE, "TF init", "echo:"]

        detected = [p.device for p in serial.tools.list_ports.comports()]
        if not detected:
            raise RuntimeError("No serial ports found. Is the printer connected?")

        info(f"Available ports: {detected}")

        for dev in detected:
            debug(f"Trying {dev} ...")
            ser, lines = _probe_port(
                port_device=dev,
                baud=baud,
                indicators=indicators,
                request=b"M115\n",
                read_window_s=10,
                min_lines=3,
            )
            if ser is not None:
                self._serial = ser
                info(f"Printer found on {dev}")
                for ln in lines[-10:]:
                    debug(f"[{dev}] {ln}")
                return
            warning(f"{dev} did not respond as a compatible printer ({len(lines)} line(s))")

        raise RuntimeError(f"Printer not found on any port. Tried: {detected}")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                cmd = self._command_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._exec(cmd)
            except (MotionFault, MotionTimeout) as exc:
                self.faulted = True
                error(f"[FAULT] {exc}")
                self._emit(f"[FAULT] {exc}", log=True)
                break
            except Exception as exc:
                self.faulted = True
                error(f"[UNHANDLED ERROR] {exc}")
                self._emit(f"[UNHANDLED ERROR] {exc}", log=True)
                break

    # ------------------------------------------------------------------
    # G-code execution
    # ------------------------------------------------------------------

    def _exec(self, cmd: _Command) -> None:
        if cmd.message:
            self._emit(cmd.message, cmd.log)

        gc = cmd.gcode.strip()
        self._track_position(gc)
        self._send_and_wait(gc)

    def _send_and_wait(self, gc: str) -> None:
        self._serial.write(f"{gc}\n".encode())
        self._await_ok()

    def _await_ok(self, deadline_s: float = 60.0) -> None:
        end = time.time() + deadline_s
        while True:
            if time.time() > end:
                raise MotionTimeout("Timed out waiting for 'ok' from printer")
            line = self._serial.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                time.sleep(0.02)
                continue
            low = line.lower()
            if low in ("ok", "processing"):
                return
            if line.lower().startswith("error"):
                raise MotionFault(f"Printer error: {line}")
            debug(f"[printer] {line}")

    def _track_position(self, gc: str) -> None:
        upper = gc.upper()
        if upper == "G28":
            self.position = Position(0, 0, 0)
            return
        cmd_code = upper.split()[0] if upper else ""
        if cmd_code not in ("G0", "G1"):
            return
        updates: dict[str, int] = {}
        for axis, pattern in [("x", r"X([\d.]+)"), ("y", r"Y([\d.]+)"), ("z", r"Z([\d.]+)")]:
            match = re.search(pattern, gc, re.IGNORECASE)
            if match:
                updates[axis] = round(float(match.group(1)) * _NM_PER_MM)
        if updates:
            self.position = Position(
                x=updates.get("x", self.position.x),
                y=updates.get("y", self.position.y),
                z=updates.get("z", self.position.z),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _home(self) -> None:
        self._exec(_Command("G90", message="Setting absolute positioning"))
        self._exec(_Command("G28", message="Homing..."))

    def _enqueue(self, gc: str, message: str = "", log: bool = False) -> None:
        if self.faulted:
            warning("Ignoring command: controller is faulted")
            return
        self._command_queue.put(_Command(gc, message or None, log))

    def _emit(self, text: str, log: bool = False) -> None:
        if log:
            info(text)
        for listener in list(self._message_listeners):
            try:
                listener(text, log)
            except Exception as exc:
                warning(f"Message listener raised: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message_listener(self, listener: Callable[[str, bool], None]) -> None:
        """Subscribe to controller messages.  Signature: (text: str, log: bool) -> None."""
        if not callable(listener):
            raise TypeError("listener must be callable")
        self._message_listeners.append(listener)

    def remove_message_listener(self, listener: Callable[[str, bool], None]) -> None:
        try:
            self._message_listeners.remove(listener)
        except ValueError:
            pass

    def get_position(self) -> Position:
        """Return the current position (coordinates in nanometres)."""
        return self.position

    def get_bed_size(self) -> Position:
        """Return the machine's maximum extents as a Position (nanometres)."""
        return Position.from_mm(self.config.max_x, self.config.max_y, self.config.max_z)

    def set_speed(self, speed_nm: int) -> None:
        """
        Set the jog step size in nanometres.

        The value is clamped to a minimum of config.step_size so the
        controller cannot be set below its hardware resolution.
        """
        self.speed = max(self.config.step_size, speed_nm)
        info(f"Speed set to {self.speed / _NM_PER_MM:.6f} mm ({self.speed} nm)")

    def move_to_position(self, position: Position) -> None:
        """Enqueue an absolute move to *position* (coordinates in nanometres)."""
        self._enqueue(
            f"G0 {position.to_gcode()}",
            message=f"Moving to {position}",
        )

    def move(self, axis: str, amount_nm: int, *, is_relative: bool = True) -> bool:
        """
        Move *axis* by *amount_nm* nanometres.

        When *is_relative* is True (the default) *amount_nm* is treated as a
        delta from the current position.  When False it is treated as an
        absolute target position in nanometres.

        Returns False (and does not enqueue) if the resulting position would
        exceed the configured axis limits.
        """
        current_nm: int = getattr(self.position, axis)
        new_nm = current_nm + amount_nm if is_relative else amount_nm

        max_nm = getattr(self.config, f"max_{axis}") * _NM_PER_MM
        if not 0 <= new_nm <= max_nm:
            return False

        mm = new_nm / _NM_PER_MM
        self._enqueue(f"G1 {axis.upper()}{mm:.6f}")
        return True

    def move_axis(self, axis: str, direction: int) -> bool:
        """
        Jog *axis* by one speed-step in *direction* (+1 or -1).

        Returns False (and does not enqueue) if the resulting position would
        exceed the configured axis limits.
        """
        return self.move(axis, self.speed * direction)

    def home(self) -> None:
        """Enqueue a homing sequence (G90 + G28)."""
        self._enqueue("G90", message="Set absolute positioning")
        self._enqueue("G28", message="Homing...")

    def reset_fault(self) -> None:
        """Clear a faulted state so the controller can accept commands again."""
        self.faulted = False

    def shutdown(self) -> None:
        """Signal the worker thread to stop and wait for it to exit."""
        self._stop_event.set()
        self._thread.join(timeout=5)
        if hasattr(self, "_serial") and self._serial.is_open:
            self._serial.close()
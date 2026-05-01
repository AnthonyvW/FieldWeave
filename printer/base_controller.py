from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import Callable, Iterable
import serial
import serial.tools.list_ports
import time
import queue
import threading
import re

from .models import Position
from .motion_config import (
    MotionSystemSettings,
    MotionSystemSettingsManager,
)

# 1 mm expressed in nanometres — mirrors the constant in models.py
_NM_PER_MM = 1_000_000


def _probe_port(port_device, baud, indicators, request=b"M115\r\n", read_window_s=10, min_lines=3):
    """
    Try to identify a Marlin-like printer on a single serial port.
    Returns (serial_connection, response_lines) on success, or (None, response_lines) on failure.
    On success, the serial connection is LEFT OPEN for the caller.
    """
    responses = []
    ser = None
    success = False
    try:
        ser = serial.Serial(
            port_device,
            baudrate=baud,
            timeout=0.25,
            write_timeout=1
        )

        # Some controllers reset on open due to DTR; give them a brief window to chatter.
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

        # Ask for firmware info
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

        if success:
            return ser, responses
        else:
            return None, responses
    except Exception:
        return None, responses
    finally:
        if ser is not None and ser.is_open and not success:
            try:
                ser.close()
            except Exception:
                pass


class PrinterFault(Exception): pass
class PrinterTimeout(Exception): pass

@dataclass
class Command:
    kind: str
    value: object
    message: str | None = None
    log: bool = False

class BasePrinterController:
    """Base class for 3D printer / motion-system control."""

    def __init__(self) -> None:
        # Load motion system config via its manager (mirrors FieldWeaveSettingsManager pattern)
        self._config_manager = MotionSystemSettingsManager()
        self.config: MotionSystemSettings = self._config_manager.load()

        # All positions stored in nanometres for sub-micron Z precision.
        self.position = Position(0, 0, 0)

        # Current jog step size in nanometres (initialised from config).
        self.speed: int = self.config.step_size

        self.faulted = False

        # _running: set means the worker is free to process commands.
        self._running = threading.Event()
        self._running.set()

        # _stop_requested: set means the current automation should abort.
        self._stop_requested = threading.Event()

        # _ready: set once _initialize_printer succeeds on the worker thread.
        self._ready = threading.Event()
        self._init_error: Exception | None = None

        self.command_queue: Queue[Command] = Queue()

        # Buffer for macros when paused.
        self._front_buffer: list[Command] = []
        self._front_lock = threading.Lock()

        # Callbacks
        self._message_listeners: list[Callable[[str, bool, Command], None]] = []
        self._handlers: dict[str, Callable[[Command], None]] = {}

        # Register built-in handlers
        self.register_handler("PRINTER", self._handle_printer)
        self.register_handler("MACRO", self._handle_macro)
        self.register_handler("MACRO_WAIT", self._handle_macro)
        self.register_handler("STATUS", self._handle_status)

        # Start worker thread — it will call _initialize_printer itself, then
        # set _ready (or store the error) before entering the command loop.
        self._processing_thread = threading.Thread(
            target=self._worker, daemon=True
        )
        self._processing_thread.start()

    def is_ready(self) -> bool:
        """Return True once the printer has been found and homed."""
        return self._ready.is_set()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """
        Block until the printer is ready (or timeout seconds have elapsed).
        Returns True if ready, False if timed out.
        Raises the initialisation exception if connection failed.
        """
        ready = self._ready.wait(timeout)
        if self._init_error is not None:
            raise self._init_error
        return ready

    def _initialize_printer(self) -> None:
        """Initialize printer serial connection."""
        baud = self.config.baud_rate
        indicators = [self.config.FIRMWARE_NAME, self.config.MACHINE_TYPE, "TF init", "echo:"]

        detected = [p.device for p in serial.tools.list_ports.comports()]
        if not detected:
            raise RuntimeError("No serial ports found. Is the printer connected?")

        print(f"Available ports: {detected}")

        for dev in detected:
            try:
                print(f"Trying port {dev}")
                ser, lines = _probe_port(
                    port_device=dev,
                    baud=baud,
                    indicators=indicators,
                    request=b"M115\n",
                    read_window_s=10,
                    min_lines=3,
                )
                if ser is not None:
                    self.printer_serial = ser
                    print(f"Printer found on port: {dev}")
                    for ln in lines[-10:]:
                        print(f"[{dev}] {ln}")
                    return
                else:
                    print(f"Port {dev} did not respond as a compatible printer. Observed {len(lines)} line(s).")
            except Exception as e:
                print(f"Port {dev} failed: {e}")

        raise RuntimeError(f"Printer not found on any available serial port. Tried: {detected}")

    def _worker(self) -> None:
        """
        Worker thread entry point.  Initialises the serial connection, homes
        the printer, signals readiness, then drives the command loop.
        """
        try:
            self._initialize_printer()
        except Exception as exc:
            self._init_error = exc
            self._ready.set()
            return

        self.home()
        self._ready.set()
        self._process_commands()

    def _process_commands(self) -> None:
        while True:
            self._running.wait()

            try:
                cmd: Command | None = None
                with self._front_lock:
                    if self._front_buffer:
                        cmd = self._front_buffer.pop(0)
                if cmd is None:
                    try:
                        cmd = self.command_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                if cmd.message:
                    self._emit_message(cmd.message, cmd.log, cmd)
                (self._handlers.get(cmd.kind) or self._handle_unknown)(cmd)

            except (PrinterFault, PrinterTimeout) as e:
                self.force_stop(f"Printer error: {e}")
            except Exception as e:
                self.force_stop(f"Unhandled error: {e}")

    def _push_front(self, cmd: Command) -> None:
        with self._front_lock:
            self._front_buffer.insert(0, cmd)

    def _exec_gcode(
        self,
        gc: str,
        *,
        wait: bool = False,
        update_position: bool | None = None,
        message: str | None = None,
        log: bool = False,
    ) -> None:
        if message:
            self._emit_message(message, log, None)

        gc_stripped = gc.strip()
        if update_position is None:
            update_position = gc_stripped.upper().startswith("G")

        if update_position:
            self._update_position(gc_stripped)

        self._send_and_wait(gc_stripped)

        if wait and gc_stripped.upper() != "M400":
            self._send_and_wait("M400")

    def register_handler(self, kind: str, fn: Callable[[Command], None]) -> None:
        self._handlers[kind] = fn

    def _handle_printer(self, cmd: Command) -> None:
        self._exec_gcode(cmd.value, wait=False, message=None, log=False)

    def _handle_unknown(self, cmd: Command) -> None:
        self._emit_message(f"Unknown kind: {cmd.kind}", True, cmd)

    def _send_and_wait(self, command: str) -> None:
        """Send command and wait for OK response."""
        self._send_command(command)
        self._wait_for_ok()

    def _wait_for_ok(self, deadline_s: float = 60.0) -> None:
        """Wait for 'ok' (or 'processing'), ignore empty reads, and time out cleanly."""
        end = time.time() + deadline_s
        while True:
            if time.time() > end:
                raise PrinterTimeout("Timed out waiting for 'ok' from printer")

            response = self.printer_serial.readline().decode("utf-8", errors="ignore").strip()
            if not response:
                time.sleep(0.02)
                continue

            low = response.lower()
            if low == "ok" or low == "processing":
                break
            if response.startswith("error"):
                raise PrinterFault(f"Printer reported error: {response}")
            print(response)

    def _update_position(self, command: str) -> None:
        """
        Update internal position tracking based on G-code movement commands.

        G-code coordinates are in millimetres; position is stored in nanometres.
        """
        upper = command.strip().upper()

        if upper == "G28":
            self.position = Position(x=0, y=0, z=0)
            return

        move_code = upper.split()[0] if upper else ""
        if move_code not in ("G0", "G1"):
            return

        updates: dict[str, int] = {}
        for axis, pattern in [('x', r'X([\d.]+)'), ('y', r'Y([\d.]+)'), ('z', r'Z([\d.]+)')]:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                mm = float(match.group(1))
                updates[axis] = round(mm * _NM_PER_MM)

        if updates:
            self.position = Position(
                x=updates.get('x', self.position.x),
                y=updates.get('y', self.position.y),
                z=updates.get('z', self.position.z),
            )

    def _send_command(self, command: str) -> None:
        """Send command to printer."""
        self.printer_serial.write(f"{command}\n".encode())

    def get_position(self) -> Position:
        """Get current position (coordinates in nanometres)."""
        return self.position

    def get_bed_size(self) -> Position:
        """Return the machine's maximum extents as a Position (nanometres)."""
        return Position.from_mm(self.config.max_x, self.config.max_y, self.config.max_z)

    def get_max_x(self) -> int:
        """Maximum X travel in millimetres."""
        return self.config.max_x

    def get_max_y(self) -> int:
        """Maximum Y travel in millimetres."""
        return self.config.max_y

    def get_max_z(self) -> int:
        """Maximum Z travel in millimetres."""
        return self.config.max_z

    def move_to_position(self, position: Position) -> None:
        """Move to specified position (coordinates in nanometres)."""
        self.enqueue_printer(f"G0 {position.to_gcode()}", message=f"Moving to {position}", log=False)

    def move_axis(self, axis: str, direction: int) -> bool:
        """
        Jog the given axis by one step in the specified direction.

        The step size (self.speed) is in nanometres.  Axis limits (config.max_*)
        are in millimetres and are converted to nanometres for comparison.

        Returns False if the resulting position would be out of bounds.
        """
        current_nm: int = getattr(self.position, axis)
        new_nm = current_nm + (self.speed * direction)

        max_mm: int = getattr(self.config, f"max_{axis}")
        max_nm = max_mm * _NM_PER_MM

        if 0 <= new_nm <= max_nm:
            mm_value = new_nm / _NM_PER_MM
            self.enqueue_printer(f"G1 {axis.upper()}{mm_value:.6f}", log=False)
            return True
        return False

    def force_stop(self, reason: str = "fault") -> None:
        self.faulted = True
        self._stop_requested.set()
        self._running.clear()
        self._flush_pipeline()
        self._emit_message(f"[FORCE STOP] {reason}", True, None)

    def reset_fault(self) -> None:
        """Clear all state after a hard (force) stop."""
        self.faulted = False
        self._stop_requested.clear()
        self._running.set()

    def home(self) -> None:
        self.enqueue_printer("G90", "Set all axes to absolute positioning")
        self.enqueue_printer("G28", "Homing printer...")

    def flush_moves(self) -> None:
        self._exec_gcode("M400", wait=True, update_position=False, message="Waiting for moves...", log=True)


    # Command Queuer

    def enqueue_cmd(self, cmd: Command) -> None:
        """Enqueue any command by kind."""
        if self.faulted:
            self._emit_message("Ignored: controller faulted; call reset_fault()", True, cmd)
            return
        self.command_queue.put(cmd)

    def create_cmd(self, kind: str, value: str, message: str = "", log: bool = False) -> Command:
        return Command(kind, value, message or None, log)

    def enqueue_printer(self, gc: str, message: str = "", log: bool = True) -> None:
        self.enqueue_cmd(self.printer_cmd(gc, message, log))

    def printer_cmd(self, gc: str, message: str = "", log: bool = False) -> Command:
        """
        Create a PRINTER command.

        Printer commands wrap raw G-code strings so they can be placed on the
        controller queue or used inside macros. The command is executed by
        sending the G-code to the printer.

        Args:
            gc: G-code string (e.g. "G28", "G0 X10 Y10").
            message: Optional message displayed when the command runs.
            log: If True, also print the message to stdout.

        Example:
            # Build a simple move command
            move = self.printer_cmd("G0 X50 Y50 Z5", message="Move to 50,50,5", log=True)

            # Enqueue immediately
            self.enqueue_cmd(move)

            # Or include it in a macro
            steps = [
                self.printer_cmd("G28", message="Home"),
                move,
            ]
            home_and_move = self.macro_cmd(steps, wait_printer=True)
            self.enqueue_cmd(home_and_move)
        """
        return Command(kind="PRINTER", value=gc, message=message or None, log=log)

    def _flush_pipeline(self) -> None:
        with self._front_lock:
            self._front_buffer.clear()
        try:
            while True:
                self.command_queue.get_nowait()
        except queue.Empty:
            pass


    # Automation

    def stop(self, reason: str = "user") -> None:
        self._stop_requested.set()
        self._running.clear()
        self._emit_message("Stopped", True, None)
        self._flush_pipeline()

    def resume(self) -> None:
        """Clear a soft stop so new commands can run."""
        self._stop_requested.clear()
        self._running.set()

    def toggle_pause(self) -> None:
        """Toggle pause state."""
        if self._running.is_set():
            self._running.clear()
            self._emit_message("Paused", True, None)
        else:
            self._running.set()
            self._emit_message("Unpaused", True, None)

    def status_cmd(self, message: str, log: bool = True) -> Command:
        return Command(kind="STATUS", value=message, message=message, log=log)

    def _handle_status(self, cmd: Command, emit: bool = False) -> None:
        if emit:
            self._emit_message(cmd.message, cmd.log, cmd)
        elif cmd.log:
            print("[Status]", cmd.message)


    # Macro Handling

    def _handle_macro(self, cmd: Command) -> None:
        steps = list(cmd.value or [])
        wait_printer = (cmd.kind.upper() == "MACRO_WAIT")

        for i, sub in enumerate(steps):
            if self._stop_requested.is_set():
                return

            if not self._running.is_set():
                remaining = steps[i:]
                if remaining and not self._stop_requested.is_set():
                    self._push_front(Command(cmd.kind, remaining, None, cmd.log))
                return

            if self.pause_point():
                return

            if sub.kind == "PRINTER":
                self._exec_gcode(
                    sub.value,
                    wait=wait_printer,
                    message=sub.message,
                    log=sub.log,
                )
            else:
                (self._handlers.get(sub.kind) or self._handle_unknown)(sub)

    def macro_cmd(
        self,
        items: Iterable[Command],
        *,
        wait_printer: bool = False,
        message: str | None = None,
        log: bool = False,
    ) -> Command:
        """
        Create a MACRO command (a batch of commands executed atomically).

        Args:
            items: Iterable of Command objects.
            wait_printer: If True, each PRINTER move waits (M400) before the next.
            message: Optional message displayed when the macro begins.
            log: If True, also print messages to stdout.
        """
        return Command(
            kind="MACRO_WAIT" if wait_printer else "MACRO",
            value=items,
            message=message,
            log=log,
        )

    def pause_point(self) -> bool:
        """
        Cooperative pause/stop point for automation loops.
        Blocks (without spinning) for as long as the controller is paused.
        Returns True if a stop has been requested, False otherwise.
        """
        self._running.wait()
        return self._stop_requested.is_set()


    # Message Listener Hooks

    def add_message_listener(self, listener: Callable[[str, bool, Command], None]) -> None:
        """Subscribe to command messages. Signature: (text, log, cmd) -> None"""
        if not callable(listener):
            raise TypeError("listener must be callable")
        self._message_listeners.append(listener)

    def remove_message_listener(self, listener: Callable[[str, bool, Command], None]) -> None:
        try:
            self._message_listeners.remove(listener)
        except ValueError:
            pass

    def _emit_message(self, text: str, log: bool, cmd: Command | None = None) -> None:
        if log and text:
            print(text)
        if text:
            for listener in list(self._message_listeners):
                try:
                    listener(text, log, cmd)
                except Exception as e:
                    print(f"[warn] message listener raised: {e}")


    # Convenience methods for movement via the GUI

    def move_z_up(self) -> None:
        self.resume()
        self.move_axis('z', 1)

    def move_z_down(self) -> None:
        self.resume()
        self.move_axis('z', -1)

    def move_x_left(self) -> None:
        self.resume()
        self.move_axis('x', 1)

    def move_x_right(self) -> None:
        self.resume()
        self.move_axis('x', -1)

    def move_y_backward(self) -> None:
        self.resume()
        self.move_axis('y', 1)

    def move_y_forward(self) -> None:
        self.resume()
        self.move_axis('y', -1)


    # Speed controls

    def adjust_speed(self, amount: int) -> None:
        """Adjust the jog step size by the given amount (nanometres)."""
        self.speed = max(self.config.step_size, self.speed + amount)
        print(f"Current step size: {self.speed / _NM_PER_MM:.6f} mm ({self.speed} nm)")

    def increase_speed(self) -> None:
        self.adjust_speed(self.config.step_size)

    def decrease_speed(self) -> None:
        self.adjust_speed(-self.config.step_size)

    def increase_speed_fast(self) -> None:
        self.adjust_speed(self.config.step_size * 25)

    def decrease_speed_fast(self) -> None:
        self.adjust_speed(-self.config.step_size * 25)
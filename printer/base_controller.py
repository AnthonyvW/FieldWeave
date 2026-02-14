
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import Optional, Callable, List, Dict, Iterable
import serial
import serial.tools.list_ports
import time
import queue
import threading
import re

from .models import Position
from .printerConfig import (
    PrinterSettings,
    PrinterSettingsManager
)
from fieldweaveConfig import (
    FieldWeaveSettings,
)

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
            timeout=0.25,       # slightly less chatty spin
            write_timeout=1
        )

        # Some controllers reset on open due to DTR; give them a brief window to chatter.
        start = time.time()
        quiet_since = start
        while time.time() - start < 2.0:   # ~2s settle (exits earlier if quiet)
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

        # Read with a firm window
        start = time.time()
        while time.time() - start < read_window_s:
            if ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    responses.append(line)
                    if any(ind in line for ind in indicators):
                        success = True
                        break
                    # Heuristic: enough lines + a few seconds → likely the right device
                    if len(responses) >= min_lines and time.time() - start > 3:
                        success = True
                        break
            else:
                time.sleep(0.05)

        if success:
            return ser, responses  # leave open
        else:
            return None, responses
    except Exception:
        return None, responses
    finally:
        # Only close when NOT successful
        if ser is not None and ser.is_open and not success:
            try:
                ser.close()
            except Exception:
                pass


class PrinterFault(Exception): pass
class PrinterTimeout(Exception): pass

@dataclass
class command:
    kind: str 
    value: any
    message: Optional[str] = None   
    log: bool = False

class BasePrinterController:
    CONFIG_SUBDIR = "Ender3"
    """Base class for 3D printer control"""
    def __init__(self, fieldweaveConfig: FieldWeaveSettings):
        self.config = PrinterSettings()
        PrinterSettingsManager.scope_dir(self.CONFIG_SUBDIR)
        self.config = PrinterSettingsManager.load(self.CONFIG_SUBDIR)

        self.position = Position(0, 0, 0) # Current position
        self.speed = self.config.step_size  # Current Speed
        self.paused = False
        self.stop_requested = False
        self.faulted = False

        self.command_queue: Queue[command] = Queue()

        # Buffer for macros when paused.
        self._front_buffer: List[command] = []
        self._front_lock = threading.Lock()
        
        # Callbacks
        self._message_listeners: List[Callable[[str, bool, command], None]] = []
        self._handlers: Dict[str, Callable[[command], None]] = {}
        
        # register built-ins
        self.register_handler("PRINTER", self._handle_printer)
        self.register_handler("MACRO", self._handle_macro)
        self.register_handler("MACRO_WAIT", self._handle_macro)
        self.register_handler("STATUS", self._handle_status)


        # Initialize serial connection
        self._initialize_printer(fieldweaveConfig)
        
        # Start command processing thread
        self._processing_thread = threading.Thread(target=self._process_commands, daemon=True)
        self._processing_thread.start()

    def _initialize_printer(self, fieldweaveConfig):
        """Initialize printer serial connection"""
        baud = self.config.baud_rate
        indicators = getattr(self.config, "valid_response_indicators", None) or [
            "FIRMWARE_NAME", "Marlin", "Ender", "TF init", "echo:"
        ]

        # Ports list: configured (first) then all others (no duplicates)
        detected = [p.device for p in serial.tools.list_ports.comports()]
        if not detected:
            raise RuntimeError("No serial ports found. Is the printer connected?")

        preferred = []
        cfg_port = getattr(fieldweaveConfig, "serial_port", None)
        if cfg_port:
            preferred = [cfg_port]
        remaining = [p for p in detected if p not in set(preferred)]
        ports_to_try = preferred + remaining

        print(f"Available ports (preferred first): {ports_to_try}")

        for dev in ports_to_try:
            try:
                label = "(configured)" if preferred and dev == preferred[0] else ""
                print(f"Trying port {dev} {label}".strip())
                ser, lines = _probe_port(
                    port_device=dev,
                    baud=baud,
                    indicators=indicators,
                    request=b"M115\n",
                    read_window_s=10,
                    min_lines=3,
                )
                if ser is not None:
                    self.printer_serial = ser  # keep open
                    print(f"Printer found on port: {dev}")
                    # show last few lines like old version
                    for ln in lines[-10:]:
                        print(f"[{dev}] {ln}")
                    return
                else:
                    print(f"Port {dev} did not respond as a compatible printer. Observed {len(lines)} line(s).")
            except Exception as e:
                print(f"Port {dev} failed: {e}")

        raise RuntimeError(f"Printer not found on any available serial port. Tried: {ports_to_try}")

    def _process_commands(self):
        time.sleep(1)
        self.home()
        while True:
            if self.paused:
                time.sleep(0.05)
                continue
            try:
                cmd = None
                with self._front_lock:
                    if self._front_buffer:
                        cmd = self._front_buffer.pop(0)
                if cmd is None:
                    cmd = self.command_queue.get()

                if cmd.message:
                    self._emit_message(cmd.message, cmd.log, cmd)
                (self._handlers.get(cmd.kind) or self._handle_unknown)(cmd)

            except (PrinterFault, PrinterTimeout) as e:
                self.halt(f"Printer error: {e}")
            except Exception as e:
                self.halt(f"Unhandled error: {e}")

    def _push_front(self, cmd: command) -> None:
        with self._front_lock:
            self._front_buffer.insert(0, cmd)

    def _exec_gcode(
        self,
        gc: str,
        *,
        wait: bool = False,                 # add M400 after sending (ensures motion complete)
        update_position: bool | None = None, # auto: update for G*; override if needed
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

        # send primary command
        self._send_and_wait(gc_stripped)

        # optionally ensure the move fully finishes at firmware level
        if wait and gc_stripped.upper() != "M400":
            self._send_and_wait("M400")

    def _parse_kv(self, s: str) -> dict[str, str]:
        out = {}
        for tok in s.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                out[k] = v
        return out

    def register_handler(self, kind: str, fn: Callable[[command], None]) -> None:
        self._handlers[kind] = fn

    def _handle_printer(self, cmd):    # default
        self._exec_gcode(cmd.value, wait=False, message=None, log=False)

    def _handle_unknown(self, cmd):    
        self._emit_message(f"Unknown kind: {cmd.kind}", True, cmd)


    def _send_and_wait(self, command: str) -> None:
        """Send command and wait for OK response"""
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
                # Nothing arrived during the serial timeout window; don't print a blank line.
                time.sleep(0.02)
                continue

            low = response.lower()
            if low == "ok" or low == "processing":
                break
            if response.startswith("error"):
                raise PrinterFault(f"Printer reported error: {response}")
            print(response)  # only non-empty, informative lines

    def _update_position(self, command: str) -> None:
        """Update internal position tracking based on G-code command"""
        # Handle full homing
        if command.strip().upper() == "G28":
            self.position = Position(x=0, y=0, z=0)
            return

        updates = {}
        for axis, pattern in [('x', r'X([\d\.]+)'), ('y', r'Y([\d\.]+)'), ('z', r'Z([\d\.]+)')]:
            match = re.search(pattern, command)
            if match:
                updates[axis] = int(float(match.group(1)) * 100)
        
        if updates:
            self.position = Position(
                x=updates.get('x', self.position.x),
                y=updates.get('y', self.position.y),
                z=updates.get('z', self.position.z)
            )

    def _send_command(self, command: str) -> None:
        """Send command to printer"""
        self.printer_serial.write(f"{command}\n".encode())

    def get_position(self) -> Position:
        """Get current position as tuple"""
        return self.position

    def get_bed_size(self)-> Position:
        return Position(self.config.max_x, self.config.max_y, self.config.max_z)
    
    def get_max_x(self)-> int:
        return self.config.max_x // 100
    
    def get_max_y(self)-> int:
        return self.config.max_y // 100
    
    def get_max_z(self)-> int:
        return self.config.max_z // 100

    def move_to_position(self, position: Position) -> None:
        """Move to specified position"""
        self.enqueue_printer(f"G0 {position.to_gcode()}", message=f"Moving to {position}", log=False)

    def move_axis(self, axis: str, direction: int) -> None:
        """Move specified axis by current speed * direction"""
        current_value = getattr(self.position, axis)
        new_value = current_value + (self.speed * direction)

        max_value = getattr(self.config, f"max_{axis}")
        if 0 <= new_value <= max_value:
            self.enqueue_printer(f"G1 {axis.upper()}{new_value / 100}", log=False)

    def force_stop(self, reason="fault"):
        self.faulted = True
        self.stop_requested = True
        self.paused = True
        self._flush_pipeline()
        self._emit_message(f"[FORCE STOP] {reason}", True, None)

    def reset_force_stop(self):
        self.faulted = False
        self.stop_requested = False
        self.paused = False

    def home(self) -> None:
        
        self.enqueue_printer("G90", "Set all Axis to Absolute Positioning")
        self.enqueue_printer("G28", "Homing Printer. . .")
            
    def flush_moves(self) -> None:
        self._exec_gcode("M400", wait=True, update_position=False, message="Waiting for moves...", log=True)





    # Command Queuer
    def enqueue_cmd(self, cmd: command) -> None:
        """Enqueue any command by kind."""
        if self.faulted:
            self._emit_message("Ignored: controller faulted; call reset()", True, cmd)
            return
        self.command_queue.put(cmd)

    def create_cmd(self, kind: str, value: str, message: str = "", log: bool = False) -> None:
        # Creates a command with the arguments
        return command(kind, value, message or None, log)

    def enqueue_printer(self, gc: str, message: str = "", log: bool = True) -> None:
        self.enqueue_cmd(self.printer_cmd(gc, message, log))

    def printer_cmd(self, gc: str, message: str = "", log: bool = False) -> command:
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
        return command(kind="PRINTER", value=gc, message = message or None, log=log)

    def _flush_pipeline(self):
        with self._front_lock:
            self._front_buffer.clear()
        try:
            while True:
                self.command_queue.get_nowait()
        except queue.Empty:
            pass


    # Automation
    def stop(self, reason="user"):
        # Soft stop: pause, mark stop, nuke anything pending/resumable
        self.stop_requested = True
        self.paused = True
        self._emit_message("Stopped", True, None)
        self._flush_pipeline()
    
    def reset_after_stop(self):
        # Clearing stop lets new commands run; nothing old will resume
        self.stop_requested = False
        self.paused = False

    def toggle_pause(self) -> None:
        """Toggle pause state"""
        self.paused = not self.paused
        self._emit_message("Unpaused" if not self.paused else "Paused", True, None)

    def status_cmd(self, message: str, log: bool = True):
        return command(kind="STATUS", value=message, message = message, log=log)

    def _handle_status(self, cmd: command, emit = False):
        if(emit):
            self._emit_message(cmd.message, cmd.log, cmd)
        elif(cmd.log):
            print("[Status]", cmd.message)


    # Macro Handling
    def _handle_macro(self, cmd: command) -> None:
        steps = list(cmd.value or [])
        wait_printer = (cmd.kind.upper() == "MACRO_WAIT")

        for i, sub in enumerate(steps):
            if self.stop_requested:
                # Abort immediately; do NOT requeue the remainder.
                return

            if self.paused:
                # Only resume later if not stopping.
                remaining = steps[i:]
                if remaining and not self.stop_requested:
                    self._push_front(command(cmd.kind, remaining, None, cmd.log))
                return

            # cooperative pause/stop point between sub-steps
            if self.pause_point():
                return  # stop requested; do not requeue

            # Execute one atomic sub-step
            if sub.kind == "PRINTER":
                self._exec_gcode(
                    sub.value,
                    wait=wait_printer,
                    message=sub.message,
                    log=sub.log,
                )
            else:
                (self._handlers.get(sub.kind) or self._handle_unknown)(sub)

    def macro_cmd(self, items: Iterable[command], *, wait_printer: bool = False, message: str | None = None, log: bool = False) -> command:
        """
        Create a MACRO command (a batch of commands executed atomically).

        Macros are useful for grouping a sequence of commands into one logical
        unit. When enqueued, the macro will expand and run its steps in order.

        Args:
            steps: Iterable of `command` objects (e.g. from printer_cmd, camera_cmd, etc.)
            wait_printer: If True, each PRINTER move waits (M400) before the next.
            message: Optional message displayed when the macro begins.
            log: If True, also print messages to stdout.

        Example:
            # Build a square-move macro
            steps = [
                self.printer_cmd("G28", message="Homing...", log=True),
                self.printer_cmd("G0 X0 Y0", message="Start corner"),
                self.printer_cmd("G0 X10 Y0", message="Move right"),
                self.printer_cmd("G0 X10 Y10", message="Move up"),
                self.printer_cmd("G0 X0 Y10", message="Move left"),
                self.printer_cmd("G0 X0 Y0", message="Back to start"),
            ]

            square_macro = self.macro_cmd(
                steps,
                wait_printer=True,
                message="Square pattern macro",
                log=True,
            )

            # Enqueue like any other command
            self.enqueue_cmd(square_macro)
        """
        return command(
            kind="MACRO_WAIT" if wait_printer else "MACRO",
            value=items,
            message=message,
            log=log,
        )

    def pause_point(self):
        """Pause points that can be inserted into automation to pause it"""
        while self.paused and not self.stop_requested:
            time.sleep(0.05)
        return self.stop_requested


    # Message Listener Hooks
    def add_message_listener(self, listener: Callable[[str, bool, command], None]) -> None:
        """Subscribe to command messages. Signature: (text, log, cmd) -> None"""
        if not callable(listener):
            raise TypeError("listener must be callable")
        self._message_listeners.append(listener)

    def remove_message_listener(self, listener: Callable[[str, bool, command], None]) -> None:
        try:
            self._message_listeners.remove(listener)
        except ValueError:
            pass  # already removed / never added

    def _emit_message(self, text: str, log: bool, cmd: Optional[command] = None) -> None:
        # Console logging only when requested
        if log and text:
            print(text)

        # Notify listeners (e.g., UI)
        if text:
            for listener in list(self._message_listeners):
                try:
                    listener(text, log, cmd)
                except Exception as e:
                    # Keep the pipeline resilient if a listener explodes
                    print(f"[warn] message listener raised: {e}")


    # Convenience methods for movement
    def move_z_up(self): 
        self.reset_after_stop()
        self.move_axis('z', 1)

    def move_z_down(self): 
        self.reset_after_stop()
        self.move_axis('z', -1)

    def move_x_left(self): 
        self.reset_after_stop()
        self.move_axis('x', 1)

    def move_x_right(self): 
        self.reset_after_stop()
        self.move_axis('x', -1)

    def move_y_backward(self): 
        self.reset_after_stop()
        self.move_axis('y', 1)

    def move_y_forward(self): 
        self.reset_after_stop()
        self.move_axis('y', -1)


    # Methods for speed
    def adjust_speed(self, amount: int) -> None:
        """Adjust movement speed"""
        self.speed = max(self.config.step_size, self.speed + amount)  # Prevent negative speed
        print("Current Speed", self.speed / 100)

    def increase_speed(self): self.adjust_speed(self.config.step_size)
    def decrease_speed(self): self.adjust_speed(-self.config.step_size)
    def increase_speed_fast(self): self.adjust_speed(self.config.step_size * 25)
    def decrease_speed_fast(self): self.adjust_speed(-self.config.step_size * 25)
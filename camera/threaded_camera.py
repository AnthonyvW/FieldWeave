"""
Threaded camera wrapper using dynamic attribute access.
Provides full IDE type hinting by transparently proxying to the underlying camera.
"""

from __future__ import annotations

from typing import Callable, Any, TypeVar, Generic
from queue import Queue, Empty
from threading import Thread, Event, Lock
from functools import wraps
import time

from PySide6.QtCore import QObject, Signal

from camera.cameras.base_camera import BaseCamera
from common.logger import info, error, warning, debug, exception

T = TypeVar('T', bound=BaseCamera)


class AsyncResult:
    """Represents the result of an async camera operation."""

    def __init__(self):
        self._event = Event()
        self._success = False
        self._result = None

    def set_result(self, success: bool, result: Any) -> None:
        self._success = success
        self._result = result
        self._event.set()

    def wait(self, timeout: float | None = None) -> tuple[bool, Any]:
        self._event.wait(timeout)
        return self._success, self._result


class CameraCommand:
    """Command to execute on the camera thread."""

    def __init__(
        self,
        method_name: str,
        args: tuple,
        kwargs: dict,
        completion_callback: Callable | None = None,
    ):
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
        self.completion_callback = completion_callback
        self.result = AsyncResult()


class ShutdownCommand:
    """Sentinel that causes the camera thread to exit its loop."""
    pass


class CameraThread(QObject):
    """
    Qt-aware camera thread that runs camera operations in background.

    Signals:
        operation_completed: Emitted when any operation completes (method_name, success, result)
        error_occurred:      Emitted when an error occurs (error_msg)
    """

    operation_completed = Signal(str, bool, object)
    error_occurred = Signal(str)

    def __init__(self, camera: BaseCamera):
        super().__init__()
        self._camera = camera
        self._command_queue: Queue = Queue()
        self._thread: Thread | None = None
        self._running = Event()
        self._lock = Lock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            warning("Camera thread already running")
            return

        self._running.set()
        self._thread = Thread(target=self._run, daemon=True, name="CameraThread")
        self._thread.start()
        info("Camera thread started")

    def stop(self, wait: bool = True) -> None:
        if self._thread is None or not self._thread.is_alive():
            return

        info("Stopping camera thread")
        self._running.clear()

        pending_count = 0
        while True:
            try:
                command = self._command_queue.get_nowait()
            except Empty:
                break
            if not isinstance(command, ShutdownCommand):
                command.result.set_result(False, None)
                pending_count += 1

        if pending_count > 0:
            info(f"Cancelled {pending_count} pending commands")

        self._command_queue.put(ShutdownCommand())

        if wait and self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                warning("Camera thread did not stop within 3 seconds")
            else:
                info("Camera thread stopped successfully")

    def execute(self, command: CameraCommand) -> AsyncResult:
        self._command_queue.put(command)
        return command.result

    def _run(self) -> None:
        debug("Camera thread running")

        while self._running.is_set():
            if self._command_queue.empty():
                time.sleep(0.05)
                continue

            try:
                command = self._command_queue.get_nowait()
            except Empty:
                continue

            if isinstance(command, ShutdownCommand):
                debug("Received shutdown command")
                break

            if not self._running.is_set():
                debug(f"Thread stopping, skipping command: {command.method_name}")
                command.result.set_result(False, None)
                continue

            success, result = self._execute_command(command)
            command.result.set_result(success, result)
            self.operation_completed.emit(command.method_name, success, result)

        debug("Camera thread exiting")

    def _execute_command(self, command: CameraCommand) -> tuple[bool, Any]:
        with self._lock:
            method = getattr(self._camera, command.method_name, None)
            if method is None:
                error(f"Method not found: {command.method_name}")
                return False, None

            try:
                result = method(*command.args, **command.kwargs)
                if isinstance(result, bool):
                    return result, None
                return True, result
            except Exception as e:
                exception(f"Error calling {command.method_name}: {e}")
                self.error_occurred.emit(str(e))
                return False, None


class ThreadedCamera(Generic[T]):
    """
    Wrapper around BaseCamera that executes all operations in a background thread.

    Uses __getattr__ to transparently proxy method calls to the underlying camera,
    keeping full IDE type hinting via the Generic[T] annotation.

    All proxied methods accept two extra keyword arguments:
        wait (bool):                  Block until the operation completes.
        on_complete (callable|None):  Called on the *main* thread (via the
                                      operation_completed signal) with
                                      (success: bool, result: Any).

    Note: on_complete is wired through the operation_completed signal so it
    always runs on the main thread regardless of which thread execute() is
    called from. Routines must never call GUI methods or emit signals directly.
    """

    def __init__(self, camera: T):
        object.__setattr__(self, '_camera', camera)
        object.__setattr__(self, '_thread', CameraThread(camera))
        object.__setattr__(self, '_started', False)
        object.__setattr__(self, '_pending_callbacks', {})

    def start_thread(self) -> None:
        self._thread.start()
        object.__setattr__(self, '_started', True)

    def stop_thread(self, wait: bool = True) -> None:
        self._thread.stop(wait)
        object.__setattr__(self, '_started', False)

    @property
    def operation_completed(self):
        return self._thread.operation_completed

    @property
    def error_occurred(self):
        return self._thread.error_occurred

    @property
    def underlying_camera(self) -> T:
        return self._camera

    def __getattr__(self, name: str):
        attr = getattr(self._camera, name)

        if not callable(attr):
            return attr

        @wraps(attr)
        def threaded_method(
            *args,
            wait: bool = False,
            on_complete: Callable[[bool, Any], None] | None = None,
            **kwargs,
        ):
            if not self._started:
                debug(f"Camera thread not running, calling {name} on main thread")
                result = attr(*args, **kwargs)
                if wait:
                    if isinstance(result, tuple):
                        return result
                    if isinstance(result, bool):
                        return result, None
                    return True, result
                return result

            command = CameraCommand(name, args, kwargs)

            if on_complete is not None:
                pending: dict[str, Callable] = object.__getattribute__(self, '_pending_callbacks')
                pending[id(command)] = on_complete

                def _dispatch(method_name: str, success: bool, cb_result: Any) -> None:
                    cmd_id = id(command)
                    cb = pending.pop(cmd_id, None)
                    if cb is not None:
                        cb(success, cb_result)

                self._thread.operation_completed.connect(_dispatch)

            async_result = self._thread.execute(command)

            if wait:
                return async_result.wait()

            return None

        return threaded_method

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ('_camera', '_thread', '_started', '_pending_callbacks'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._camera, name, value)

    def __dir__(self):
        return list(set(
            dir(type(self)) +
            list(self.__dict__.keys()) +
            dir(self._camera)
        ))


def create_threaded_camera(camera: T) -> T:
    """
    Factory function to create a ThreadedCamera typed as the underlying camera class,
    giving full IDE autocomplete without any cast at the call site.
    """
    return ThreadedCamera(camera)  # type: ignore
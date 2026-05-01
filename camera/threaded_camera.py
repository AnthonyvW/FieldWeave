"""
Threaded camera wrapper using dynamic attribute access.
Provides full IDE type hinting by transparently proxying to the underlying camera.
"""

from typing import Optional, Callable, Any, TypeVar, Generic
from queue import Queue, Empty
from threading import Thread, Event, Lock
from functools import wraps
import time

from PySide6.QtCore import QObject, Signal

from camera.cameras.base_camera import BaseCamera
from common.logger import info, error, warning, debug, exception

T = TypeVar('T', bound=BaseCamera)


class AsyncResult:
    """
    Represents the result of an async operation.
    Can be used with callbacks or awaited in the future.
    """
    def __init__(self):
        self._event = Event()
        self._success = False
        self._result = None
    
    def set_result(self, success: bool, result: Any):
        self._success = success
        self._result = result
        self._event.set()
    
    def wait(self, timeout: Optional[float] = None) -> tuple[bool, Any]:
        """Wait for result (blocking)"""
        self._event.wait(timeout)
        return self._success, self._result


class CameraCommand:
    """Command to execute on camera thread"""
    def __init__(
        self,
        method_name: str,
        args: tuple,
        kwargs: dict,
        completion_callback: Optional[Callable] = None
    ):
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
        self.completion_callback = completion_callback
        self.result = AsyncResult()


class ShutdownCommand:
    """Signal to shutdown the thread"""
    pass


class CameraThread(QObject):
    """
    Qt-aware camera thread that runs camera operations in background.
    
    Signals:
        operation_completed: Emitted when any operation completes (method_name, success, result)
        error_occurred: Emitted when an error occurs (error_msg)
    """
    
    operation_completed = Signal(str, bool, object)  # method_name, success, result
    error_occurred = Signal(str)  # error_msg
    
    def __init__(self, camera: BaseCamera):
        super().__init__()
        self._camera = camera
        self._command_queue = Queue()
        self._thread = None
        self._running = Event()
        self._lock = Lock()
        
    def start(self):
        """Start the camera thread"""
        if self._thread is not None and self._thread.is_alive():
            warning("Camera thread already running")
            return
            
        self._running.set()
        self._thread = Thread(target=self._run, daemon=True, name="CameraThread")
        self._thread.start()
        info("Camera thread started")
    
    def stop(self, wait: bool = True):
        """Stop the camera thread"""
        if self._thread is None or not self._thread.is_alive():
            return
            
        info("Stopping camera thread")
        self._running.clear()
        
        # Clear any pending commands
        pending_count = 0
        try:
            while True:
                command = self._command_queue.get_nowait()
                if not isinstance(command, ShutdownCommand):
                    # Signal that this command won't be executed
                    command.result.set_result(False, None)
                    if command.completion_callback:
                        try:
                            command.completion_callback(False, None)
                        except:
                            pass
                    pending_count += 1
        except Empty:
            pass
        
        if pending_count > 0:
            info(f"Cancelled {pending_count} pending commands")
        
        # Send shutdown command
        self._command_queue.put(ShutdownCommand())
        
        if wait and self._thread is not None:
            # Wait longer for thread to finish processing
            self._thread.join(timeout=3.0)  # Reduced from 10s
            if self._thread.is_alive():
                warning("Camera thread did not stop within 3 seconds")
            else:
                info("Camera thread stopped successfully")
    
    def execute(self, command: CameraCommand) -> AsyncResult:
        """
        Execute a command and return AsyncResult
        
        Args:
            command: The command to execute
            
        Returns:
            AsyncResult that can be waited on or ignored
        """
        self._command_queue.put(command)
        return command.result
    
    def _run(self):
        """Main thread loop"""
        debug("Camera thread running")
        
        while self._running.is_set():
            try:
                # Get command with timeout
                if self._command_queue.empty():
                    time.sleep(0.05)
                    continue
                try:
                    command = self._command_queue.get_nowait()
                except Exception:
                    continue
                
                # Handle shutdown
                if isinstance(command, ShutdownCommand):
                    debug("Received shutdown command")
                    break
                
                # Check if we should still process (thread might be stopping)
                if not self._running.is_set():
                    debug(f"Thread stopping, skipping command: {command.method_name}")
                    command.result.set_result(False, None)
                    if command.completion_callback:
                        try:
                            command.completion_callback(False, None)
                        except Exception as e:
                            exception(f"Error in completion callback: {e}")
                    continue
                
                # Execute command
                try:
                    success, result = self._execute_command(command)
                    
                    # Set result
                    command.result.set_result(success, result)
                    
                    # Emit signal
                    self.operation_completed.emit(command.method_name, success, result)
                    
                    # Call completion callback if provided
                    if command.completion_callback is not None:
                        try:
                            command.completion_callback(success, result)
                        except Exception as e:
                            exception(f"Error in completion callback: {e}")
                    
                except Exception as e:
                    exception(f"Error executing {command.method_name}")
                    error_msg = str(e)
                    
                    command.result.set_result(False, None)
                    self.operation_completed.emit(command.method_name, False, None)
                    self.error_occurred.emit(error_msg)
                    
                    if command.completion_callback is not None:
                        try:
                            command.completion_callback(False, None)
                        except Exception as cb_error:
                            exception(f"Error in completion callback: {cb_error}")
                    
            except Exception as e:
                exception("Unexpected error in camera thread")
                self.error_occurred.emit(str(e))
        
        debug("Camera thread exiting")
    
    def _execute_command(self, command: CameraCommand) -> tuple[bool, Any]:
        """Execute a single command"""
        with self._lock:
            method = getattr(self._camera, command.method_name, None)
            if method is None:
                error(f"Method not found: {command.method_name}")
                return False, None
            
            try:
                result = method(*command.args, **command.kwargs)
                
                # If method returns bool, use that as success indicator
                # Otherwise assume success
                if isinstance(result, bool):
                    return result, None
                else:
                    return True, result
                    
            except Exception as e:
                exception(f"Error calling {command.method_name}")
                raise


class ThreadedCamera(Generic[T]):
    """
    Wrapper around BaseCamera that executes all operations in a background thread.
    
    This class uses Python's __getattr__ magic method to transparently proxy
    all method calls to the underlying camera, providing full IDE type hinting.
    
    Usage:
        # Create with type hint for full IDE support
        base_camera = AmscopeCamera()
        camera: AmscopeCamera = ThreadedCamera(base_camera)
        camera.start_thread()
        
        # Now you get full type hinting and autocomplete!
        camera.set_white_balance(5000, 1000)  # IDE knows this method exists
        camera.auto_white_balance()           # IDE autocompletes this
        
        # All methods are async by default
        camera.snap_image(0)  # Returns immediately
        
        # Use callbacks for chaining
        camera.snap_image(0, on_complete=lambda s, r: print("Done!"))
        
        # Or wait for result
        success, result = camera.snap_image(0, wait=True)
    """
    
    def __init__(self, camera: T):
        # Use object.__setattr__ to avoid triggering __setattr__
        object.__setattr__(self, '_camera', camera)
        object.__setattr__(self, '_thread', CameraThread(camera))
        object.__setattr__(self, '_started', False)
    
    def start_thread(self):
        """Start the background thread"""
        self._thread.start()
        object.__setattr__(self, '_started', True)
    
    def stop_thread(self, wait: bool = True):
        """Stop the background thread"""
        self._thread.stop(wait)
        object.__setattr__(self, '_started', False)
    
    @property
    def operation_completed(self):
        """Access to operation_completed signal"""
        return self._thread.operation_completed
    
    @property
    def error_occurred(self):
        """Access to error_occurred signal"""
        return self._thread.error_occurred
    
    @property
    def underlying_camera(self) -> T:
        """Get the underlying camera instance"""
        return self._camera
    
    def __getattr__(self, name: str):
        """
        Magic method that intercepts all attribute access.
        
        This provides transparent proxying to the underlying camera while
        running everything in the background thread.
        """
        # Get the attribute from underlying camera
        attr = getattr(self._camera, name)
        
        # If it's not callable, just return it
        if not callable(attr):
            return attr
        
        # If it's a method, wrap it
        @wraps(attr)
        def threaded_method(
            *args,
            wait: bool = False,
            on_complete: Optional[Callable[[bool, Any], None]] = None,
            **kwargs
        ):
            """
            Threaded wrapper for camera methods.
            
            Args:
                *args: Positional arguments for the method
                wait: If True, wait for operation to complete (blocking)
                on_complete: Optional callback(success, result) when done
                **kwargs: Keyword arguments for the method
                
            Returns:
                If wait=True: (success, result)
                If wait=False: None
            """
            if not self._started:
                debug(f"Camera thread not running, calling {name} on main thread")
                # Call underlying method directly
                result = attr(*args, **kwargs)
                
                # If wait=True, we need to return a tuple
                # But underlying method might return bool, None, or tuple
                if wait:
                    if isinstance(result, tuple):
                        return result
                    elif isinstance(result, bool):
                        return (result, None)
                    else:
                        # None or other - treat as success
                        return (True, result)
                return result
            
            # Create command
            command = CameraCommand(name, args, kwargs, on_complete)
            
            # Execute
            result = self._thread.execute(command)
            
            # Wait if requested
            if wait:
                return result.wait()
            
            return None
        
        return threaded_method
    
    def __setattr__(self, name: str, value: Any):
        """
        Intercept attribute setting to forward to underlying camera.
        """
        # Our own attributes (those set in __init__)
        if name in ('_camera', '_thread', '_started'):
            object.__setattr__(self, name, value)
        else:
            # Forward to underlying camera
            setattr(self._camera, name, value)
    
    def __dir__(self):
        """
        Return the combined attributes of this class and the underlying camera.
        This helps IDE autocomplete work properly.
        """
        return list(set(
            dir(type(self)) +
            list(self.__dict__.keys()) +
            dir(self._camera)
        ))


def create_threaded_camera(camera: T) -> T:
    """
    Factory function to create a threaded camera with proper type hints.
    
    Args:
        camera: The base camera instance
        
    Returns:
        ThreadedCamera that appears as the same type as input
        
    Example:
        base = AmscopeCamera()
        camera = create_threaded_camera(base)  # Type is AmscopeCamera
        camera.set_white_balance(5000, 1000)   # Full type hints!
    """
    return ThreadedCamera(camera)  # type: ignore
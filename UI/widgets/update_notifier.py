from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from common.app_context import get_app_context
from common.updater import Updater, UpdateStatus, relaunch


class UpdateNotifier(QWidget):
    """
    Polls the Updater on a QTimer and drives all update UI from the main
    thread. The Updater's background thread only ever mutates plain data -
    this widget is the only place that reads that data and touches Qt.
    """

    def __init__(self, updater: Updater, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.hide()

        self._updater = updater
        self._prompted = False
        self._manual = False
        self._updating_dialog: QMessageBox | None = None

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(500)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start()

        self._updater.start_check()

    def check_for_updates(self, manual: bool = False) -> bool:
        """
        Trigger a check. When manual=True (e.g. an About-page button), the
        result is shown as a popup instead of a toast. Returns False if busy.
        """
        self._prompted = False
        self._manual = manual
        if not self._poll_timer.isActive():
            self._poll_timer.start()
        started = self._updater.start_check()
        if not started:
            self._manual = False
        return started

    def _poll(self) -> None:
        status = self._updater.status

        if status == UpdateStatus.UPDATE_AVAILABLE and not self._prompted:
            self._prompted = True
            # Stop polling before the modal Yes/No prompt below: QMessageBox.question()
            # runs a nested event loop, and a still-running poll_timer would keep firing
            # _poll() reentrantly for as long as the user leaves the prompt open.
            self._poll_timer.stop()
            self._prompt_update()

        elif status == UpdateStatus.UP_TO_DATE:
            self._poll_timer.stop()
            self._notify_up_to_date()
            self._manual = False

        elif status == UpdateStatus.CHECK_FAILED:
            self._poll_timer.stop()
            self._notify_check_failed()
            self._manual = False

        elif status == UpdateStatus.NO_UPSTREAM:
            self._poll_timer.stop()
            self._notify_no_upstream()
            self._manual = False

        elif status == UpdateStatus.UPDATING and self._updating_dialog is None:
            self._show_updating_dialog()

        elif status == UpdateStatus.UPDATE_COMPLETE:
            self._poll_timer.stop()
            self._close_updating_dialog()
            self._show_restart_message()

        elif status == UpdateStatus.UPDATE_FAILED:
            self._poll_timer.stop()
            self._close_updating_dialog()
            self._show_update_failed_message()

    def _prompt_update(self) -> None:
        behind = self._updater.commits_behind
        plural = "s" if behind != 1 else ""

        reply = QMessageBox.question(
            self.parentWidget(),
            "Update Available",
            f"FieldWeave is {behind} commit{plural} behind. Update now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._updater.start_update()
            self._poll_timer.start()

    def _show_restart_message(self) -> None:
        reply = QMessageBox.question(
            self.parentWidget(),
            "Update Complete",
            "FieldWeave has been updated. Restart now to apply it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._restart_now()

    def _show_update_failed_message(self) -> None:
        QMessageBox.warning(
            self.parentWidget(),
            "Update Failed",
            f"The update could not be completed:\n{self._updater.error_message}",
        )

    def _show_updating_dialog(self) -> None:
        # Non-modal show() rather than exec() - exec() would start a nested
        # event loop and block this widget's own poll_timer from firing.
        # WindowModal (not ApplicationModal) - combining ApplicationModal
        # with show() rather than exec() can leave Qt's modal grab stuck
        # active even after the dialog is closed, freezing the whole app.
        dialog = QMessageBox(self.parentWidget())
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setWindowTitle("Updating")
        dialog.setText("FieldWeave is updating. Do not close the program.")
        dialog.setStandardButtons(QMessageBox.StandardButton.NoButton)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)
        dialog.show()
        self._updating_dialog = dialog

    def _close_updating_dialog(self) -> None:
        if self._updating_dialog is not None:
            # done() is the proper QDialog teardown (releases the modal grab
            # and emits finished()); close() alone can leave it half-released.
            self._updating_dialog.done(0)
            self._updating_dialog = None
            # Force the close to be fully processed now, so the dialog is
            # actually gone from screen before the next modal dialog opens.
            QApplication.processEvents()

    def _restart_now(self) -> None:
        # Release camera/motion hardware before the process image is replaced,
        # since execv skips normal Qt/Python shutdown and finalizers.
        get_app_context().cleanup()
        relaunch()

    def _toast_success(self, message: str) -> None:
        toast = get_app_context().toast
        if toast is not None:
            toast.success(message)

    def _toast_warning(self, message: str) -> None:
        toast = get_app_context().toast
        if toast is not None:
            toast.warning(message)

    def _notify_up_to_date(self) -> None:
        if self._manual:
            QMessageBox.information(
                self.parentWidget(),
                "No Updates Available",
                "FieldWeave is up to date.",
            )
        else:
            self._toast_success("FieldWeave is up to date")

    def _notify_no_upstream(self) -> None:
        if self._manual:
            QMessageBox.information(
                self.parentWidget(),
                "Update Check Skipped",
                "The current branch is not tracking a remote, so update checking is unavailable.",
            )

    def _notify_check_failed(self) -> None:
        message = f"Failed to check for updates: {self._updater.error_message}"
        if self._manual:
            QMessageBox.warning(self.parentWidget(), "Update Check Failed", message)
        else:
            self._toast_warning(message)
# Camera Preview Regression Checklist

Manual verification checklist for the camera preview / measurement refactor
(see the interaction-mode, input-tool-dispatch, and measurement-kind-registry
work tracked against this branch). Run through this after any change to
`UI/widgets/camera_preview.py`, `UI/widgets/preview_overlay/`, or the tabs
that drive preview interaction mode, before considering that change done.

## Navigate tab
- [ ] Click on the live feed moves the stage (once machine vision is
      calibrated and motion is ready).
- [ ] Click does nothing while uncalibrated, or while an automation routine
      is running.
- [ ] Plain mouse wheel scroll moves the Z axis; Ctrl+scroll zooms instead.
- [ ] `+`/`=`/`-` zoom in/out; arrow keys pan while zoomed, and fall through
      to normal focus navigation while not zoomed.

## Measurement tab
- [ ] Click-to-move is fully suppressed (live feed and loaded image alike).
- [ ] Plain scroll zooms (same as Ctrl+scroll); there is no Z-move path.
- [ ] Selecting a measurement kind in the sidebar, then clicking once,
      starts a draft; a dashed preview line/shape follows the cursor
      between clicks with no button held.
- [ ] Completing the required number of clicks finalizes the measurement
      and it renders with its tag/label.
- [ ] Placing a measurement while zoomed in resolves to the correct
      underlying frame position (not the zoomed/panned widget position).
- [ ] Click-and-drag while a kind is selected pans the zoomed view instead
      of placing a measurement (drag threshold ~4px).
- [ ] Clicking directly on an existing measurement's endpoint (not
      currently placing a new one) starts an endpoint drag; dragging moves
      just that point, respecting kind-specific constraints (e.g.
      Horizontal Line keeps its row fixed).
- [ ] Right-click while a multi-point placement is in progress cancels the
      draft (Arbitrary Line instead finalizes if >=2 points already placed).
- [ ] Hovering a measurement's tag shows/highlights it without a prior
      click; a delete glyph appears and removing it deletes the
      measurement.
- [ ] Pressing Delete/Backspace while hovering a tag deletes that
      measurement, with no click required first.
- [ ] Clicking a tag (not the delete glyph) opens the customize menu;
      editing there live-previews on the placed measurement; Apply commits,
      Cancel reverts.
- [ ] Switching unit (e.g. mm to px) updates all displayed labels.
- [ ] Manual DPI calibration: starting it, clicking a two-point line,
      entering a real-world length, submitting, and cancelling all behave
      as today.
- [ ] Switching to "loaded image" mode shows the static image, measurements
      persist per-image, and switching images with existing measurements
      prompts a discard confirmation.
- [ ] Leaving the tab (switching to Navigate/Calibration/Project) restores
      click-to-move and Z-scroll on return to Navigate; measurement overlay
      is hidden while another tab is showing the shared preview.

## Calibration wizards
- [ ] Camera Space Calibration: crosshair shows on the appropriate step,
      restores to its prior state when leaving the wizard, and the final
      "verify" step's click-to-move still works (default preview behavior,
      no wizard-specific code).
- [ ] DPI Calibration: crosshair on step 1, inspect-calibration overlay on
      steps 2/3, both off afterward; tick-length slider updates the overlay
      live; overlay state (crosshair/inspect-calibration) restores to
      whatever it was before entering the wizard when finishing or
      resetting.
- [ ] Slot Calibration: red-mark overlay shows only on its designated
      steps and restores to its prior state afterward.
- [ ] Re-entering a wizard after finishing it once (via the launch list)
      resets its step to 1 and re-applies step-1 overlay state.

## Cross-cutting
- [ ] Switching tabs while zoomed in a measurement/loaded-image context
      does not leave zoom/pan state corrupted when returning.
- [ ] No overlay state "leaks" between tabs (e.g. crosshair left on after
      leaving a calibration wizard when it wasn't on beforehand).

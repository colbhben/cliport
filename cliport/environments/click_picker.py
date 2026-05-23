"""Mouse-click object picker for PyBullet GUI.

The CLIPort environment exposes a PyBullet GUI when constructed with
`disp=True`. This module wraps PyBullet's mouse-event API to let a human
operator click on an object in the rendered 3D scene; the click is converted
to a world-space ray (via `getDebugVisualizerCamera`) and ray-cast against
the scene to identify which object was clicked.

Usage:
    picker = ClickPicker(allowed_ids=task.block_ids)
    clicked_id, attempts, click_ms_relative = picker.wait_for_click(
        prompt="click the red block",
        correct_id=task.target_block_id,
        clock_fn=lambda: logger.relative_ms(),
    )

`wait_for_click` returns only after the user clicks one of the
`allowed_ids` AND that click matches `correct_id`. Wrong clicks are
re-prompted (printed to stdout) and dropped on the floor (per project
design). The returned `click_ms_relative` is the value of `clock_fn()`
sampled at the moment of the accepted click — pass `Logger.relative_ms`
so the click timestamp shares t=0 with `data.csv`.
"""

import time

import pybullet as p


class ClickPicker:
    """Blocks until the user clicks an allowed object in the GUI."""

    def __init__(self, allowed_ids=None, poll_hz: float = 60.0):
        self.allowed_ids = set(allowed_ids) if allowed_ids is not None else None
        self._poll_dt = 1.0 / poll_hz

    # ------------------------------------------------------------------
    # Internal: convert (mouseX, mouseY) to a world-space ray.
    # ------------------------------------------------------------------

    @staticmethod
    def _screen_to_ray(mouse_x, mouse_y):
        """Return (ray_from_world, ray_to_world) for a screen-space click.

        Uses the current debug visualizer camera. Ray length 100m is more
        than enough for a CLIPort tabletop.
        """
        cam = p.getDebugVisualizerCamera()
        width, height = cam[0], cam[1]
        view_matrix = cam[2]
        proj_matrix = cam[3]
        cam_forward = cam[5]
        cam_up = cam[7]
        # PyBullet ships a helper for this in its examples; reimplemented
        # here to avoid depending on pybullet_data layout.
        # Convert NDC [-1, 1] -> world.
        ndc_x = (2.0 * mouse_x) / width - 1.0
        ndc_y = 1.0 - (2.0 * mouse_y) / height

        import numpy as np
        proj = np.array(proj_matrix).reshape(4, 4, order='F')
        view = np.array(view_matrix).reshape(4, 4, order='F')
        inv = np.linalg.inv(proj @ view)

        near = inv @ np.array([ndc_x, ndc_y, -1.0, 1.0])
        far = inv @ np.array([ndc_x, ndc_y, 1.0, 1.0])
        near = near[:3] / near[3]
        far = far[:3] / far[3]
        return tuple(near.tolist()), tuple(far.tolist())

    # ------------------------------------------------------------------
    # Public: blocking wait for an accepted click.
    # ------------------------------------------------------------------

    def wait_for_click(self, prompt: str, correct_id: int,
                       clock_fn=lambda: time.monotonic() * 1000.0):
        """Block until the user left-clicks the correct object.

        Args:
            prompt: shown via `print` and `p.addUserDebugText`.
            correct_id: only this object id is accepted as a "correct" click.
                Clicks on other allowed objects are treated as wrong and the
                user is re-prompted.
            clock_fn: callable returning the timestamp (ms) to record at the
                accepted click moment. Defaults to monotonic-wall ms; pass
                `Logger.relative_ms` to share t=0 with data.csv.

        Returns:
            (clicked_id, attempts, click_ms): clicked_id == correct_id,
            attempts is the number of clicks (>=1) made before the accepted
            one (so attempts == 1 means the first click was correct),
            click_ms is `clock_fn()` sampled at the accepted click.
        """
        print(f"[click] {prompt}")
        text_uid = p.addUserDebugText(
            prompt, [0.5, 0.0, 0.5], textColorRGB=[1, 1, 1], textSize=1.5,
        )
        attempts = 0
        try:
            while True:
                events = p.getMouseEvents()
                for e in events:
                    # Event tuple: (eventType, mousePosX, mousePosY,
                    # buttonIndex, buttonState).
                    event_type = e[0]
                    button = e[3]
                    state = e[4]
                    # 2 == MOUSE_BUTTON_EVENT, button 0 == left, state & 3
                    # KEY_WAS_TRIGGERED bit set on press-down.
                    is_press = (event_type == 2 and button == 0
                                and (state & p.KEY_WAS_TRIGGERED))
                    if not is_press:
                        continue
                    mouse_x, mouse_y = e[1], e[2]
                    ray_from, ray_to = self._screen_to_ray(mouse_x, mouse_y)
                    hits = p.rayTest(ray_from, ray_to)
                    if not hits:
                        continue
                    obj_id = hits[0][0]
                    if obj_id < 0:
                        continue
                    if (self.allowed_ids is not None
                            and obj_id not in self.allowed_ids):
                        attempts += 1
                        print(f"[click] miss (id={obj_id} not allowed); try again")
                        continue
                    attempts += 1
                    if obj_id != correct_id:
                        print(f"[click] wrong object (id={obj_id}); try again")
                        continue
                    click_ms = float(clock_fn())
                    return obj_id, attempts, click_ms
                time.sleep(self._poll_dt)
        finally:
            p.removeUserDebugItem(text_uid)

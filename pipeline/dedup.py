"""
Decides whether a candidate event is worth paying Claude to judge.

Measured on real footage, the trigger produced 86 events an hour. Judging all
of them costs about $11 per camera per 12-hour day and, worse, turns into
interruptions a person learns to ignore. Alert fatigue - not accuracy - is what
kills systems like this: thirty notifications a shift and the owner mutes it,
at which point detection quality is irrelevant.

Two rules, deliberately:

- A tracked person is judged once per cooldown window. One person loitering is
  one event, not forty.
- A hard floor between judged events on a camera, whoever is in frame. Tracker
  ids churn badly on low-resolution footage - one real clip produced 21 ids for
  5 people - so the per-track rule alone would leak. This is the backstop.

Suppressed events are still RECORDED. The owner sees everything in the
dashboard; what suppression controls is what costs money and what interrupts.
"""
import threading

from . import config


class Suppressor:
    def __init__(self, track_cooldown: float = None, camera_cooldown: float = None):
        self.track_cooldown = (config.DEDUP_TRACK_COOLDOWN_SECONDS
                               if track_cooldown is None else track_cooldown)
        self.camera_cooldown = (config.DEDUP_CAMERA_COOLDOWN_SECONDS
                                if camera_cooldown is None else camera_cooldown)
        self._last_track = {}    # (camera_id, track_id) -> event time
        self._last_camera = {}   # camera_id -> event time
        self._lock = threading.Lock()
        self.suppressed = 0
        self.allowed = 0

    def check(self, camera_id: str, track_ids, event_time: float):
        """
        Returns (judge: bool, reason: str). `event_time` is seconds on the
        source's own clock, so a file scan and a live camera behave the same.
        """
        camera_id = camera_id or "_"
        track_ids = set(track_ids or ())

        with self._lock:
            last_cam = self._last_camera.get(camera_id)
            if last_cam is not None and event_time - last_cam < self.camera_cooldown:
                self.suppressed += 1
                return False, (f"camera cooldown: {event_time - last_cam:.0f}s since the last "
                               f"judged event, floor is {self.camera_cooldown:.0f}s")

            if track_ids:
                seen = [t for t in track_ids
                        if (ts := self._last_track.get((camera_id, t))) is not None
                        and event_time - ts < self.track_cooldown]
                # only suppress when EVERY person here was judged recently; a
                # new person joining the scene is a new thing to look at
                if seen and len(seen) == len(track_ids):
                    self.suppressed += 1
                    return False, (f"all {len(track_ids)} tracked people judged within the last "
                                   f"{self.track_cooldown:.0f}s")

            self._last_camera[camera_id] = event_time
            for t in track_ids:
                self._last_track[(camera_id, t)] = event_time
            self.allowed += 1
            return True, "ok"

    def stats(self) -> dict:
        with self._lock:
            return {"judged": self.allowed, "suppressed": self.suppressed,
                    "tracks_seen": len(self._last_track)}

from __future__ import annotations

"""
AudioAlerter: plays spoken alerts via espeak-ng when faces are recognized.
Runs in a daemon thread so it never blocks the async event loop.
"""

import logging
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

# Minimum seconds between audio alerts per identity key
AUDIO_THROTTLE = 300  # 5 minutes


class AudioAlerter:
    def __init__(self) -> None:
        self._last_played: dict[str, float] = {}
        self._lock = threading.Lock()
        self._available = self._check_espeak()

    def _check_espeak(self) -> bool:
        try:
            subprocess.run(
                ["espeak-ng", "--version"],
                capture_output=True,
                timeout=2,
            )
            logger.info("AudioAlerter: espeak-ng available")
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("AudioAlerter: espeak-ng not found — audio alerts disabled")
            return False

    def speak(self, identity_key: str, text: str) -> None:
        """Fire-and-forget: play text if throttle allows."""
        if not self._available:
            return
        now = time.monotonic()
        with self._lock:
            if now - self._last_played.get(identity_key, 0) < AUDIO_THROTTLE:
                return
            self._last_played[identity_key] = now
        threading.Thread(target=self._play, args=(text,), daemon=True).start()

    def _play(self, text: str) -> None:
        try:
            subprocess.run(
                ["espeak-ng", "-s", "130", "-a", "200", text],
                timeout=15,
                capture_output=True,
            )
        except Exception:
            logger.exception("AudioAlerter: playback failed")

    # -- Convenience methods ------------------------------------------------

    def alert_family(self, name: str) -> None:
        self.speak(f"family:{name}", f"Hello {name}! Welcome home.")

    def alert_known(self, name: str) -> None:
        """For recognized non-family (e.g. delivery driver)."""
        self.speak(f"known:{name}", f"Hello {name}.")

    def alert_unknown(self) -> None:
        self.speak("unknown", "Hello. You are being recorded.")

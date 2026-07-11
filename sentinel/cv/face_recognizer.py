from __future__ import annotations

"""
FaceRecognizer: wraps InsightFace for face detection + embedding extraction.

Gracefully degrades to a no-op if insightface / onnxruntime is not installed.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.50  # cosine similarity; tune upward for stricter matching

try:
    from insightface.app import FaceAnalysis  # type: ignore
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _INSIGHTFACE_AVAILABLE = False
    logger.warning("insightface not installed — face recognition disabled")


class FaceRecognizer:
    """
    Manages face embeddings for enrolled persons and recognizes faces in frames.

    Usage:
        recognizer = FaceRecognizer()
        recognizer.load_embeddings([(person_id, name, role, embedding_bytes), ...])
        person_id, name, role, confidence = recognizer.recognize(frame_crop)
    """

    def __init__(self) -> None:
        self._available = _INSIGHTFACE_AVAILABLE
        self._app = None
        # (person_id, name, role, embedding_ndarray)
        self._store: list[tuple[str, str, str, np.ndarray]] = []

        if self._available:
            try:
                self._app = FaceAnalysis(
                    name="buffalo_sc",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                self._app.prepare(ctx_id=0, det_size=(320, 320))
                logger.info("FaceRecognizer ready (buffalo_sc model)")
            except Exception:
                logger.exception("Failed to initialize FaceAnalysis")
                self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def load_embeddings(
        self, rows: list[tuple[str, str, str, bytes]]
    ) -> None:
        """
        Load embeddings from the database.
        rows: list of (person_id, name, role, embedding_bytes)
        """
        self._store = [
            (pid, name, role, np.frombuffer(emb, dtype=np.float32))
            for pid, name, role, emb in rows
        ]
        logger.info("Loaded %d face embeddings", len(self._store))

    def extract_embedding(self, image: np.ndarray) -> np.ndarray | None:
        """Extract a normalized face embedding from an image crop. Returns None if no face found."""
        if not self._available or self._app is None:
            return None
        try:
            faces = self._app.get(image)
            if not faces:
                return None
            # Pick the largest detected face
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            return face.normed_embedding.astype(np.float32)
        except Exception:
            logger.exception("Error extracting face embedding")
            return None

    def recognize(
        self, image: np.ndarray
    ) -> tuple[str | None, str | None, str | None, float]:
        """
        Identify who is in the image crop.
        Returns (person_id, name, role, confidence).
        confidence is the cosine similarity (0-1); 0 if unrecognized or unavailable.
        """
        if not self._available or not self._store:
            return None, None, None, 0.0

        embedding = self.extract_embedding(image)
        if embedding is None:
            return None, None, None, 0.0

        best_pid: str | None = None
        best_name: str | None = None
        best_role: str | None = None
        best_sim = -1.0

        for pid, name, role, stored in self._store:
            sim = float(np.dot(embedding, stored))
            if sim > best_sim:
                best_sim = sim
                best_pid = pid
                best_name = name
                best_role = role

        if best_sim >= SIMILARITY_THRESHOLD:
            return best_pid, best_name, best_role, best_sim
        return None, None, None, best_sim

    def _match_embedding(
        self, embedding: np.ndarray
    ) -> tuple[str | None, str | None, str | None, float]:
        """Match a pre-computed embedding against the store. Returns (person_id, name, role, confidence)."""
        if not self._store:
            return None, None, None, 0.0

        best_pid: str | None = None
        best_name: str | None = None
        best_role: str | None = None
        best_sim = -1.0

        for pid, name, role, stored in self._store:
            sim = float(np.dot(embedding, stored))
            if sim > best_sim:
                best_sim = sim
                best_pid = pid
                best_name = name
                best_role = role

        if best_sim >= SIMILARITY_THRESHOLD:
            return best_pid, best_name, best_role, best_sim
        return None, None, None, max(best_sim, 0.0)

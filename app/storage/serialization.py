import numpy as np


def vector_to_blob(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes(order="C")


def blob_to_vector(blob: bytes) -> list[float]:
    return np.frombuffer(blob, dtype=np.float32).astype(float).tolist()

import os
from .base import StorageBackend

class LocalStorage(StorageBackend):
    def __init__(self, root_path: str):
        self.root_path = root_path

    def read(self, path: str) -> bytes:
        with open(path, 'rb') as f:
            return f.read()

    def write(self, path: str, data: bytes):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def join(self, *paths) -> str:
        return os.path.join(self.root_path, *paths)

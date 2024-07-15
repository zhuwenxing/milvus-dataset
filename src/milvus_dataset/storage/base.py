from abc import ABC, abstractmethod
import json

class StorageBackend(ABC):
    @staticmethod
    def create(root_path: str, storage_options: dict = None):
        if root_path.startswith('s3://'):
            from .s3 import S3Storage
            return S3Storage(root_path, storage_options)
        else:
            from .local import LocalStorage
            return LocalStorage(root_path)

    @abstractmethod
    def read(self, path: str) -> bytes:
        pass

    @abstractmethod
    def write(self, path: str, data: bytes):
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        pass

    @abstractmethod
    def join(self, *paths) -> str:
        pass

    def read_json(self, path: str) -> dict:
        return json.loads(self.read(path).decode('utf-8'))

    def write_json(self, path: str, data: dict):
        self.write(path, json.dumps(data).encode('utf-8'))

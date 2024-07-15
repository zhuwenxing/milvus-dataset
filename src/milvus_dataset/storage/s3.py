import boto3
from botocore.exceptions import ClientError
from .base import StorageBackend

class S3Storage(StorageBackend):
    def __init__(self, root_path: str, storage_options: dict):
        self.bucket = root_path.split('//')[1].split('/')[0]
        self.prefix = '/'.join(root_path.split('//')[1].split('/')[1:])
        self.client = boto3.client('s3', **storage_options)

    def read(self, path: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=self._get_full_path(path))
        return response['Body'].read()

    def write(self, path: str, data: bytes):
        self.client.put_object(Bucket=self.bucket, Key=self._get_full_path(path), Body=data)

    def exists(self, path: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._get_full_path(path))
            return True
        except ClientError:
            return False

    def join(self, *paths) -> str:
        return '/'.join([self.prefix] + list(paths))

    def _get_full_path(self, path: str) -> str:
        return self.join(path.lstrip('/'))

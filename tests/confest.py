import pytest


def pytest_addoption(parser):
    parser.addoption("--milvus-uri", default="http://localhost:19530", help="Milvus uri")
    parser.addoption("--milvus-token", default="root:Milvus", help="Milvus token")
    parser.addoption("--minio-endpoint-url", default="http://localhost:9000", help="MinIO endpoint url")
    parser.addoption("--minio-bucket-name", default="milvus-bucket", help="MinIO bucket name")
    parser.addoption("--minio-access-key", default="minioadmin", help="MinIO access key")
    parser.addoption("--minio-secret-key", default="minioadmin", help="MinIO secret key")


@pytest.fixture(scope="session")
def milvus_uri(request):
    """Fixture to provide Milvus uri configuration."""
    return request.config.getoption("--milvus-uri")

@pytest.fixture(scope="session")
def milvus_token(request):
    """Fixture to provide Milvus token configuration."""
    return request.config.getoption("--milvus-token")

@pytest.fixture(scope="session")
def minio_endpoint_url(request):
    """Fixture to provide MinIO endpoint url configuration."""
    return request.config.getoption("--minio-endpoint-url")

@pytest.fixture(scope="session")
def minio_bucket_name(request):
    """Fixture to provide MinIO bucket name configuration."""
    return request.config.getoption("--minio-bucket-name")

@pytest.fixture(scope="session")
def minio_access_key(request):
    """Fixture to provide MinIO access key."""
    return request.config.getoption("--minio-access-key")

@pytest.fixture(scope="session")
def minio_secret_key(request):
    """Fixture to provide MinIO secret key."""
    return request.config.getoption("--minio-secret-key")


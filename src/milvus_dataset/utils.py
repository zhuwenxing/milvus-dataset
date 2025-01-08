import json
import logging
import os
import random
import shutil
import subprocess
from pathlib import Path

import dotenv
import numpy as np
from faker import Faker
from ml_dtypes import bfloat16
from modelscope import HubApi
from pymilvus import DataType, FunctionType

dotenv.load_dotenv()


fake = Faker()
RNG = np.random.default_rng()

# Set up logger
logger = logging.getLogger(__name__)

DEFAULT_FLOAT_INDEX_PARAM = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {"M": 48, "efConstruction": 500},
}
DEFAULT_BINARY_INDEX_PARAM = {
    "index_type": "BIN_IVF_FLAT",
    "metric_type": "JACCARD",
    "params": {"M": 48},
}
DEFAULT_SPARSE_INDEX_PARAM = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
DEFAULT_BM25_INDEX_PARAM = {
    "index_type": "SPARSE_INVERTED_INDEX",
    "metric_type": "BM25",
    "params": {"bm25_k1": 1.25, "bm25_b": 0.75},
}


def gen_varchar_data(length: int, nb: int, text_mode=False):
    if text_mode:
        return [fake.text() for _ in range(nb)]
    else:
        return ["".join([chr(random.randint(97, 122)) for _ in range(length)]) for _ in range(nb)]


def gen_sparse_vectors(nb, dim=1000):
    # sparse format is dok, dict of keys

    rng = np.random.default_rng()
    vectors = [
        {d: rng.random() for d in list({*random.sample(range(dim), random.randint(20, 30)), 0, 1})}
        for _ in range(nb)
    ]
    vectors = [json.dumps(vector) for vector in vectors]
    return vectors


def gen_float_vectors(nb, dim):
    vectors = [[random.random() for _ in range(dim)] for _ in range(nb)]
    fp32_vectors = [np.array(x, dtype=np.dtype("float32")) for x in vectors]
    return fp32_vectors


def gen_binary_vectors(nb, dim):
    # binary: each int presents 8 dimension
    # so if binary vector dimension is 16, use [x, y], which x and y could be any int between 0 and 255
    vectors = [[random.randint(0, 255) for _ in range(dim // 8)] for _ in range(nb)]
    vectors = [np.array(x, dtype=np.dtype("uint8")) for x in vectors]
    return vectors


def gen_fp16_vectors(nb, dim):
    """
    generate float16 vector data
    raw_vectors : the vectors
    fp16_vectors: the bytes used for insert
    return: raw_vectors and fp16_vectors
    """
    fp16_vectors = []
    for _ in range(nb):
        raw_vector = [random.random() for _ in range(dim)]
        fp16_vector = np.array(raw_vector, dtype=np.float16).view(np.uint8).tolist()
        fp16_vectors.append(fp16_vector)
    fp16_vectors = [np.array(x, dtype=np.dtype("uint8")) for x in fp16_vectors]
    return fp16_vectors


def gen_bf16_vectors(nb, dim):
    """
    generate brain float16 vector data
    raw_vectors : the vectors
    bf16_vectors: the bytes used for insert
    return: raw_vectors and bf16_vectors
    """
    bf16_vectors = []
    for _ in range(nb):
        raw_vector = [random.random() for _ in range(dim)]
        bf16_vector = np.array(raw_vector, dtype=bfloat16).view(np.uint8).tolist()
        bf16_vectors.append(bf16_vector)
    bf16_vectors = [np.array(x, dtype=np.dtype("uint8")) for x in bf16_vectors]
    return bf16_vectors


def gen_row_data_by_schema(nb=3000, schema=None, start=None):
    if schema is None:
        raise Exception("schema is None")
    # ignore auto id field and the fields in function output
    func_output_fields = []
    if hasattr(schema, "functions"):
        functions = schema.functions
        for func in functions:
            output_field_names = func.output_field_names
            func_output_fields.extend(output_field_names)
    func_output_fields = list(set(func_output_fields))
    fields = schema.fields
    fields_needs_data = []
    for field in fields:
        if field.auto_id:
            continue
        if field.name in func_output_fields:
            continue
        fields_needs_data.append(field)
    data = []
    for _ in range(nb):
        tmp = {}
        for field in fields_needs_data:
            tmp[field.name] = gen_data_by_collection_field(field)
            if start is not None and field.dtype == DataType.INT64:
                tmp[field.name] = start
                start += 1
        data.append(tmp)
    return data


def gen_data_by_collection_field(field, nb=None, start=None):  # noqa
    # if nb is None, return one data, else return a list of data
    data_type = field.dtype
    enable_analyzer = field.params.get("enable_analyzer", False)
    if data_type == DataType.BOOL:
        if nb is None:
            return random.choice([True, False])
        return [random.choice([True, False]) for _ in range(nb)]
    if data_type == DataType.INT8:
        if nb is None:
            return random.randint(-128, 127)
        return [random.randint(-128, 127) for _ in range(nb)]
    if data_type == DataType.INT16:
        if nb is None:
            return random.randint(-32768, 32767)
        return [random.randint(-32768, 32767) for _ in range(nb)]
    if data_type == DataType.INT32:
        if nb is None:
            return random.randint(-2147483648, 2147483647)
        return [random.randint(-2147483648, 2147483647) for _ in range(nb)]
    if data_type == DataType.INT64:
        if nb is None:
            return random.randint(-9223372036854775808, 9223372036854775807)
        if start is not None:
            return list(range(start, start + nb))
        return [random.randint(-9223372036854775808, 9223372036854775807) for _ in range(nb)]
    if data_type == DataType.FLOAT:
        if nb is None:
            return np.float32(random.random())
        return [np.float32(random.random()) for _ in range(nb)]
    if data_type == DataType.DOUBLE:
        if nb is None:
            return np.float64(random.random())
        return [np.float64(random.random()) for _ in range(nb)]
    if data_type == DataType.VARCHAR:
        max_length = field.params["max_length"]
        max_length = min(20, max_length - 1)
        length = random.randint(0, max_length)
        if nb is None:
            return gen_varchar_data(length=length, nb=1, text_mode=enable_analyzer)[0]
        return gen_varchar_data(length=length, nb=nb, text_mode=enable_analyzer)
    if data_type == DataType.JSON:
        if nb is None:
            return json.dumps({"name": fake.name(), "address": fake.address()})
        data = [json.dumps({"name": str(i), "address": i} for i in range(nb))]
        return data
    if data_type == DataType.FLOAT_VECTOR:
        dim = field.params["dim"]
        if nb is None:
            vector = gen_float_vectors(nb=1, dim=dim)[0]
            vector = np.array(vector, dtype=np.float32)
            return vector
        vectors = gen_float_vectors(nb, dim)
        vectors = [np.array(vector, dtype=np.float32) for vector in vectors]
        return vectors
    if data_type == DataType.BFLOAT16_VECTOR:
        dim = field.params["dim"]
        if nb is None:
            vector = gen_bf16_vectors(nb=1, dim=dim)[0]
            return vector
        vectors = gen_bf16_vectors(nb, dim)
        return vectors
    if data_type == DataType.FLOAT16_VECTOR:
        dim = field.params["dim"]
        if nb is None:
            vector = gen_fp16_vectors(nb=1, dim=dim)[0]
            return vector
        vectors = gen_fp16_vectors(nb, dim)
        return vectors
    if data_type == DataType.BINARY_VECTOR:
        dim = field.params["dim"]
        if nb is None:
            vector = gen_binary_vectors(nb=1, dim=dim)[0]
            return vector
        vectors = gen_binary_vectors(nb, dim)
        return vectors
    if data_type == DataType.SPARSE_FLOAT_VECTOR:
        if nb is None:
            return gen_sparse_vectors(nb=1)[0]
        return gen_sparse_vectors(nb=nb)
    if data_type == DataType.ARRAY:
        max_capacity = field.params["max_capacity"]
        max_capacity = min(20, max_capacity - 1)
        element_type = field.element_type
        if element_type == DataType.INT8:
            if nb is None:
                return [random.randint(-128, 127) for _ in range(max_capacity)]
            return [[random.randint(-128, 127) for _ in range(max_capacity)] for _ in range(nb)]
        if element_type == DataType.INT16:
            if nb is None:
                return [random.randint(-32768, 32767) for _ in range(max_capacity)]
            return [[random.randint(-32768, 32767) for _ in range(max_capacity)] for _ in range(nb)]
        if element_type == DataType.INT32:
            if nb is None:
                return [random.randint(-2147483648, 2147483647) for _ in range(max_capacity)]
            return [
                [random.randint(-2147483648, 2147483647) for _ in range(max_capacity)]
                for _ in range(nb)
            ]
        if element_type == DataType.INT64:
            if nb is None:
                return [
                    random.randint(-9223372036854775808, 9223372036854775807)
                    for _ in range(max_capacity)
                ]
            return [
                [
                    random.randint(-9223372036854775808, 9223372036854775807)
                    for _ in range(max_capacity)
                ]
                for _ in range(nb)
            ]

        if element_type == DataType.BOOL:
            if nb is None:
                return [random.choice([True, False]) for _ in range(max_capacity)]
            return [[random.choice([True, False]) for _ in range(max_capacity)] for _ in range(nb)]

        if element_type == DataType.FLOAT:
            if nb is None:
                return [np.float32(random.random()) for _ in range(max_capacity)]
            return [[np.float32(random.random()) for _ in range(max_capacity)] for _ in range(nb)]
        if element_type == DataType.DOUBLE:
            if nb is None:
                return [np.float64(random.random()) for _ in range(max_capacity)]
            return [[np.float64(random.random()) for _ in range(max_capacity)] for _ in range(nb)]

        if element_type == DataType.VARCHAR:
            max_length = field.params["max_length"]
            max_length = min(20, max_length - 1)
            length = random.randint(0, max_length)
            if nb is None:
                return [
                    "".join([chr(random.randint(97, 122)) for _ in range(length)])
                    for _ in range(max_capacity)
                ]
            return [
                [
                    "".join([chr(random.randint(97, 122)) for _ in range(length)])
                    for _ in range(max_capacity)
                ]
                for _ in range(nb)
            ]
    return None


def get_scalar_field_name_list(schema=None):
    vec_fields = []
    if schema is None:
        raise Exception("schema is None")
    fields = schema.fields
    for field in fields:
        if field.dtype in [
            DataType.BOOL,
            DataType.INT8,
            DataType.INT16,
            DataType.INT32,
            DataType.INT64,
            DataType.FLOAT,
            DataType.DOUBLE,
            DataType.VARCHAR,
        ]:
            vec_fields.append(field.name)
    return vec_fields


def get_float_vec_field_name_list(schema=None):
    vec_fields = []
    if schema is None:
        raise Exception("schema is None")
    fields = schema.fields
    for field in fields:
        if field.dtype in [
            DataType.FLOAT_VECTOR,
            DataType.FLOAT16_VECTOR,
            DataType.BFLOAT16_VECTOR,
        ]:
            vec_fields.append(field.name)
    return vec_fields


def get_float16_vec_field_name_list(schema=None):
    vec_fields = []
    if schema is None:
        raise Exception("schema is None")
    fields = schema.fields
    for field in fields:
        if field.dtype == DataType.FLOAT16_VECTOR:
            vec_fields.append(field.name)
    return vec_fields


def get_bfloat16_vec_field_name_list(schema=None):
    vec_fields = []
    if schema is None:
        raise Exception("schema is None")
    fields = schema.fields
    for field in fields:
        if field.dtype == DataType.BFLOAT16_VECTOR:
            vec_fields.append(field.name)
    return vec_fields


def get_data_type_by_field_name(schema=None, field_name=None):
    if schema is None:
        raise Exception("schema is None")
    if field_name is None:
        raise Exception("field_name is None")
    fields = schema.fields
    for field in fields:
        if field.name == field_name:
            return field.dtype
    return None


def get_dim_by_field_name(schema=None, field_name=None):
    if schema is None:
        raise Exception("schema is None")
    if field_name is None:
        raise Exception("field_name is None")
    fields = schema.fields
    for field in fields:
        if field.name == field_name:
            return field.params["dim"]
    return None


def get_json_field_name_list(schema=None):
    json_fields = []
    if schema is None:
        raise Exception("schema is None")
    fields = schema.fields
    for field in fields:
        if field.dtype == DataType.JSON or (
            field.dtype == DataType.ARRAY and field.element_type == DataType.JSON
        ):
            json_fields.append(field.name)
    return json_fields


def get_binary_vec_field_name_list(schema=None):
    vec_fields = []
    if schema is None:
        raise Exception("schema is None")
    fields = schema.fields
    for field in fields:
        if field.dtype in [DataType.BINARY_VECTOR]:
            vec_fields.append(field.name)
    return vec_fields


def get_bm25_vec_field_name_list(schema=None):
    if schema is None:
        raise Exception("schema is None")
    if not hasattr(schema, "functions"):
        return []
    functions = schema.functions
    bm25_func = [func for func in functions if func.type == FunctionType.BM25]
    bm25_outputs = []
    for func in bm25_func:
        bm25_outputs.extend(func.output_field_names)
    bm25_outputs = list(set(bm25_outputs))

    return bm25_outputs


def get_sparse_vec_field_name_list(schema=None):
    vec_fields = []
    if schema is None:
        raise Exception("schema is None")
    fields = schema.fields
    for field in fields:
        if (
            field.dtype == DataType.SPARSE_FLOAT_VECTOR
            and field.name not in get_bm25_vec_field_name_list(schema)
        ):
            vec_fields.append(field.name)
    return vec_fields


def create_index_for_all_vector_fields(collection):
    schema = collection.schema
    indexes = [index.to_dict() for index in collection.indexes]
    indexed_fields = [index["field"] for index in indexes]
    float_vector_field_names = get_float_vec_field_name_list(schema)
    binary_vector_field_names = get_binary_vec_field_name_list(schema)
    sparse_vector_field_names = get_sparse_vec_field_name_list(schema)
    bm25_sparse_field_names = get_bm25_vec_field_name_list(schema)
    # create index for float vector fields
    for f in float_vector_field_names:
        if f in indexed_fields:
            continue
        collection.create_index(
            f,
            DEFAULT_FLOAT_INDEX_PARAM,
        )
    # create index for binary vector fields
    for f in binary_vector_field_names:
        if f in indexed_fields:
            continue
        collection.create_index(f, DEFAULT_BINARY_INDEX_PARAM)

    for f in sparse_vector_field_names:
        if f in indexed_fields:
            continue
        collection.create_index(f, DEFAULT_SPARSE_INDEX_PARAM)

    for f in bm25_sparse_field_names:
        if f in indexed_fields:
            continue
        collection.create_index(f, DEFAULT_BM25_INDEX_PARAM)


class ModelScopeDatasetUploader:
    def __init__(self, repo_path: str):
        """
        Initialize the dataset uploader

        Args:
            repo_path (str): ModelScope repository path in format: username/repository
                Example: wxzhuyeah/auto-create
        """
        # Validate repository path format
        if "/" not in repo_path:
            raise ValueError("Repository path should be in format: username/repository")

        self.namespace, self.dataset_name = repo_path.split("/")
        self.api = HubApi()

        # Get tokens and login
        self.sdk_token = os.getenv("MODELSCOPE_SDK_TOKEN")
        self.git_token = os.getenv("MODELSCOPE_GIT_TOKEN")
        self.api.login(access_token=self.sdk_token)

        # Build complete repository URL
        self.repo_url = (
            f"https://oauth2:{self.git_token}@www.modelscope.cn/datasets/{repo_path}.git"
        )
        self.logger = logger

        # Set working directory
        self.work_dir = Path.home() / ".modelscope"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Working directory: {self.work_dir}")

    def _filter_sensitive_info(self, text: str) -> str:
        """
        Filter sensitive information from string

        Args:
            text: Text to be filtered

        Returns:
            Filtered text
        """
        # Sensitive information patterns to be filtered
        sensitive_patterns = [
            (f"oauth2:{self.git_token}@", "oauth2:***@"),  # git token
            (self.sdk_token, "***"),  # sdk token
            (self.git_token, "***"),  # git token alone
        ]

        filtered_text = text
        for pattern, replacement in sensitive_patterns:
            if pattern:  # Ensure pattern is not empty
                filtered_text = filtered_text.replace(pattern, replacement)
        return filtered_text

    def _run_command(self, command: str | list[str], cwd: str) -> bool:
        """
        Execute shell command

        Args:
            command: Command to execute, can be string or list
            cwd: Working directory for command execution

        Returns:
            bool: Whether command execution was successful
        """
        if isinstance(command, list):
            command_str = " ".join(command)
        else:
            command_str = command

        # Filter sensitive information from logs
        safe_command = self._filter_sensitive_info(command_str)
        self.logger.info(f"Executing command: {safe_command} (in directory: {cwd})")

        try:
            with subprocess.Popen(
                command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd
            ) as p:
                stdout, stderr = p.communicate()
                stdout_str = stdout.decode("utf-8")
                stderr_str = stderr.decode("utf-8")

                # Filter sensitive information from output
                safe_stdout = self._filter_sensitive_info(stdout_str)
                safe_stderr = self._filter_sensitive_info(stderr_str)

                # Special handling for git command output
                if command_str.startswith("git"):
                    # Special status outputs for git commands
                    warning_git_messages = [
                        "nothing to commit",
                        "working tree clean",
                        "up to date",
                        "Already up to date",
                    ]

                    # Special handling for git commit
                    if command_str.startswith("git commit") and any(
                        msg in stdout_str for msg in warning_git_messages
                    ):
                        self.logger.warning(f"Git commit status: {safe_stdout.strip()}")
                        return True
                    # Handle other git commands
                    elif p.returncode != 0:
                        if any(msg in stdout_str for msg in warning_git_messages):
                            self.logger.warning(f"Git status: {safe_stdout.strip()}")
                            return True
                        else:
                            self.logger.error(
                                f"Git command failed: {safe_command}\nstdout: {safe_stdout}\nstderr: {safe_stderr}"
                            )
                            return False
                    else:
                        self.logger.info(f"Git command succeeded: {safe_stdout.strip()}")
                        return True
                else:
                    # Regular handling for non-git commands
                    if p.returncode != 0:
                        self.logger.error(
                            f"Command failed: {safe_command}\nstdout: {safe_stdout}\nstderr: {safe_stderr}"
                        )
                        return False
                    return True
        except Exception as e:
            self.logger.error(f"Error executing command: {safe_command}\nError message: {e!s}")
            return False

    def _check_git_lfs_installed(self) -> bool:
        """Check if Git LFS is installed"""
        try:
            subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            self.logger.error("Git LFS not installed. Please install Git LFS first")
            return False

    def _setup_git_lfs(self) -> bool:
        """
        Set up Git LFS and configure tracking for specific directories

        Returns:
            bool: Whether setup was successful
        """
        try:
            # Check if Git LFS is installed
            if not self._check_git_lfs_installed():
                return False

            # Initialize Git LFS
            if not self._run_command("git lfs install", self.work_dir):
                return False

            # Configure LFS tracking rules
            lfs_track_patterns = ["train/**/*", "test/**/*", "neighbors/**/*"]

            for pattern in lfs_track_patterns:
                if not self._run_command(f"git lfs track {pattern}", self.work_dir):
                    self.logger.error(f"Failed to configure Git LFS tracking: {pattern}")
                    return False
                self.logger.info(f"Configured Git LFS tracking: {pattern}")

            # Ensure .gitattributes file is added to version control
            if not self._run_command("git add .gitattributes", self.work_dir):
                return False

            return True
        except Exception as e:
            self.logger.error(f"Error setting up Git LFS: {e!s}")
            return False

    def _ensure_dataset_exists(self) -> bool:
        """
        Ensure dataset repository exists, create if it doesn't

        Returns:
            bool: Whether operation was successful
        """
        try:
            self.api.create_dataset(dataset_name=self.dataset_name, namespace=self.namespace)
            self.logger.info("Dataset created successfully")
            return True
        except Exception as e:
            # Parse error response
            error_str = str(e)
            if "Code': 10020101001" in error_str or "Name already registered" in error_str:
                # Ignore if dataset already exists
                self.logger.info(
                    f"Dataset {self.namespace}/{self.dataset_name} already exists, continuing upload process"
                )
                return True
            else:
                self.logger.error(f"Failed to create dataset: {e}")
                return False

    def _clean_work_dir(self):
        """Clean working directory while preserving .git folder"""
        work_dir = Path(self.work_dir)
        for item in work_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir() and item.name != ".git":
                shutil.rmtree(item)
        self.logger.info("Working directory cleaned")

    def _copy_files_to_temp(self, src_path: str | Path, temp_dir: str | Path) -> bool:
        """
        Copy files from source directory to temporary directory

        Args:
            src_path: Source file or directory path
            temp_dir: Temporary directory path

        Returns:
            bool: Whether copy was successful
        """
        try:
            src_path = Path(src_path)
            temp_dir = Path(temp_dir)

            if src_path.is_file():
                shutil.copy2(src_path, temp_dir)
                self.logger.info(f"Copied file: {src_path.name}")
            elif src_path.is_dir():
                for item in src_path.rglob("*"):
                    if item.is_file():
                        relative_path = item.relative_to(src_path)
                        dest_path = temp_dir / relative_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dest_path)
                        self.logger.info(f"Copied file: {relative_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to copy files: {e}")
            return False

    def upload(self, src_path: str | Path, commit_message: str | None = None) -> tuple[bool, str]:  # noqa C901
        """
        Upload file or directory to repository

        Args:
            src_path: Path to file or directory to upload
            commit_message: Commit message

        Returns:
            tuple[bool, str]: (success status, error message)
            - If successful, returns (True, "")
            - If failed, returns (False, error message)
        """
        src_path = Path(src_path)
        if not src_path.exists():
            return False, f"Path does not exist: {src_path}"

        # Ensure dataset exists
        if not self._ensure_dataset_exists():
            return False, "Failed to create or validate dataset"

        try:
            # Clean working directory
            dataset_dir = self.work_dir / self.dataset_name
            dataset_dir.parent.mkdir(parents=True, exist_ok=True)
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            self.logger.info("Cleanup complete, starting upload process...")

            # Ensure directory is clean before cloning
            self._clean_work_dir()

            # Remove .git directory (if exists) to ensure clean clone
            git_dir = self.work_dir / ".git"
            if git_dir.exists():
                shutil.rmtree(git_dir)
                self.logger.info("Removed old .git directory")

            # Clone repository
            self.logger.info("Cloning repository...")
            if not self._run_command(f"git clone {self.repo_url} .", self.work_dir):
                return False, "Failed to clone repository"

            # Set up Git LFS
            self.logger.info("Configuring Git LFS...")
            if not self._setup_git_lfs():
                return False, "Failed to configure Git LFS"

            # Copy files to working directory
            self.logger.info("Copying files to working directory...")
            if not self._copy_files_to_temp(src_path, self.work_dir):
                return False, "Failed to copy files to working directory"

            # Git operations
            git_commands = [
                ("git add -A", "Failed to add files to Git"),
                (
                    f"git commit -m \"{commit_message or 'Add dataset files'}\"",
                    "Failed to commit changes",
                ),
                ("git push origin master", "Failed to push to remote repository"),
            ]

            for cmd, error_msg in git_commands:
                if not self._run_command(cmd, self.work_dir):
                    return False, error_msg

            self.logger.info("Dataset upload completed")
            return True, ""
        except Exception as e:
            error_msg = f"Error during upload process: {e!s}"
            self.logger.error(error_msg)
            return False, error_msg
        finally:
            # Clean temporary files while preserving .git directory
            self._clean_work_dir()

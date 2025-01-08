from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymilvus import CollectionSchema

from milvus_dataset import (
    ConfigManager,
    DatasetDict,
    StorageConfig,
    StorageType,
    list_datasets,
    load_dataset,
)

app = FastAPI(
    title="Milvus Dataset API",
    description="RESTFul API for managing Milvus datasets",
    version="1.0.0",
)


class StorageConfigModel(BaseModel):
    root_path: str = "/data/datasets"
    storage_type: str = "LOCAL"
    options: dict | None = None


class DatasetGenerateModel(BaseModel):
    num_rows: int | dict[str, int]
    splits: list[str] | None = None
    target_file_size_mb: int = 512
    num_buffers: int = 15
    queue_size: int = 30
    batch_size: int = 100_000


class WriteDataModel(BaseModel):
    data: list[dict[str, Any]]
    mode: str = "append"
    writer_options: dict[str, Any] | None = None


class MilvusConfigModel(BaseModel):
    collection_name: str | None = None
    mode: str = "import"
    milvus_config: dict[str, Any]
    milvus_storage: dict[str, Any] | None = None


class StorageDestinationModel(BaseModel):
    destination: StorageConfig


class HuggingFaceConfigModel(BaseModel):
    repo_id: str
    token: str | None = None
    private: bool = False


class CreateDatasetModel(BaseModel):
    name: str
    schema: dict[str, Any]


@app.post("/init_storage")
async def init_storage(config: StorageConfigModel):
    try:
        config_manager = ConfigManager()
        config_manager.init_storage(
            root_path=config.root_path,
            storage_type=StorageType[config.storage_type],
            options=config.options,
        )
        return {"message": "Storage initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/datasets")
async def get_datasets():
    try:
        datasets = list_datasets()
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/datasets")
async def create_dataset(params: CreateDatasetModel):
    try:
        schema = CollectionSchema.construct(**params.schema)
        dataset = load_dataset(params.name, schema=schema)
        return {
            "message": f"Dataset '{params.name}' created successfully",
            "name": params.name,
            "schema": schema.to_dict(),
            "dataset": dataset.summary(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/datasets/{name}")
async def get_dataset(name: str, split: str | None = None):
    try:
        dataset = load_dataset(name, split=split)
        if isinstance(dataset, DatasetDict):
            return {
                "name": dataset.name,
                "splits": list(dataset.datasets.keys()),
                "summary": dataset.summary(),
            }
        else:
            return {"name": dataset.name, "split": dataset.split, "summary": dataset.summary()}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/datasets/{name}/write")
async def write_to_dataset(name: str, data: WriteDataModel, split: str = "train"):
    try:
        dataset = load_dataset(name, split=split)
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]

        writer = dataset.get_writer(mode=data.mode, **(data.writer_options or {}))

        for record in data.data:
            writer.write(record)
        writer.close()

        return {"message": f"Data written successfully to dataset '{name}' split '{split}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/datasets/{name}/read")
async def read_dataset(name: str, split: str = "train", mode: str = "full", batch_size: int = 1000):
    try:
        dataset = load_dataset(name, split=split)
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]

        data = []
        for batch in dataset.read(mode=mode, batch_size=batch_size):
            data.extend(batch)

        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/datasets/{name}/generate")
async def generate_dataset_data(name: str, params: DatasetGenerateModel):
    try:
        dataset = load_dataset(name)
        if not isinstance(dataset, DatasetDict):
            raise HTTPException(status_code=400, detail="Dataset must be a DatasetDict")

        dataset.generate_data(
            num_rows=params.num_rows,
            splits=params.splits,
            target_file_size_mb=params.target_file_size_mb,
            num_buffers=params.num_buffers,
            queue_size=params.queue_size,
            batch_size=params.batch_size,
        )
        return {"message": f"Data generated successfully for dataset {name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/datasets/{name}/to_milvus")
async def dataset_to_milvus(name: str, config: MilvusConfigModel):
    try:
        dataset = load_dataset(name)
        if not isinstance(dataset, DatasetDict):
            raise HTTPException(status_code=400, detail="Dataset must be a DatasetDict")

        dataset.to_milvus(
            milvus_config=config.milvus_config,
            collection_name=config.collection_name,
            mode=config.mode,
            milvus_storage=config.milvus_storage,
        )
        return {"message": f"Dataset '{name}' successfully exported to Milvus"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/datasets/{name}/to_storage")
async def dataset_to_storage(name: str, config: StorageDestinationModel):
    try:
        dataset = load_dataset(name)
        if not isinstance(dataset, DatasetDict):
            raise HTTPException(status_code=400, detail="Dataset must be a DatasetDict")

        dataset.to_storage(destination=config.destination)
        return {"message": f"Dataset '{name}' successfully copied to new storage location"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/datasets/{name}/to_huggingface")
async def dataset_to_huggingface(name: str, config: HuggingFaceConfigModel):
    try:
        dataset = load_dataset(name)
        if not isinstance(dataset, DatasetDict):
            raise HTTPException(status_code=400, detail="Dataset must be a DatasetDict")

        # Convert to HuggingFace dataset
        from datasets import Dataset as HFDataset

        hf_datasets = {}
        for split, ds in dataset.datasets.items():
            data = []
            for batch in ds.read(mode="full"):
                data.extend(batch)
            hf_datasets[split] = HFDataset.from_dict(
                {k: [d[k] for d in data] for k in data[0].keys()}
            )

        # Push to hub
        from datasets import DatasetDict as HFDatasetDict

        hf_dataset_dict = HFDatasetDict(hf_datasets)
        hf_dataset_dict.push_to_hub(
            repo_id=config.repo_id, token=config.token, private=config.private
        )

        return {"message": f"Dataset '{name}' successfully pushed to HuggingFace Hub"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/datasets/{name}/metadata")
async def get_dataset_metadata(name: str):
    try:
        dataset = load_dataset(name)
        if isinstance(dataset, DatasetDict):
            return dataset.get_metadata()
        raise HTTPException(status_code=400, detail="Dataset must be a DatasetDict")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

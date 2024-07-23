from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
import pandas as pd
from ..logging import logger
from typing import Dict, Any, Optional, Union, List
class MilvusOperations:
    def __init__(self, dataset):
        self.dataset = dataset

    def to_milvus(self, host: str = "localhost", port: str = "19530", collection_name: str = None,
                  id_field: str = None, vector_field: str = None, index=None):
        """
        Transfer the dataset to a Milvus collection, automatically generating the schema from the DataFrame.

        Args:
            collection_name (str): Name of the Milvus collection to create or use
            host (str): Milvus server host
            port (str): Milvus server port
            id_field (str): Name of the field to use as the primary key (optional)
            vector_field (str): Name of the field containing the vector data (optional)

        Returns:
            None
        """
        if collection_name is None:
            collection_name = self.dataset.name
        logger.info(f"Transferring dataset '{self.dataset.name}' to Milvus collection '{collection_name}'")

        # Connect to Milvus
        connections.connect(host=host, port=port)

        # Check if collection exists, if not, create it
        if not utility.has_collection(collection_name):
            schema = self.dataset.get_schema()
            collection = Collection(name=collection_name, schema=schema)
            logger.info(f"Created new Milvus collection: {collection_name}")
        else:
            collection = Collection(name=collection_name)
            logger.info(f"Using existing Milvus collection: {collection_name}")

        # Insert data
        for batch in self.dataset.train.read(mode='stream'):
            collection.insert(batch)

        # Flush the collection to ensure all data is written
        collection.flush()
        if index is None:
            index = {
                "index_type": "FLAT",
                "metric_type": "L2",
                "params": {},
            }
        collection.create_index(vector_field, index)

        # Load the collection for search
        collection.load()
        logger.info(f"Successfully transferred dataset '{self.dataset.name}' to Milvus collection '{collection_name}'")


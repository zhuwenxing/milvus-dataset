from pydantic import BaseModel, create_model
from typing import Dict, Any

class DatasetSchema:
    def __init__(self, fields_schema: Dict[str, Any]):
        self.fields_schema = fields_schema
        self.pydantic_model = self._create_pydantic_model()

    def _create_pydantic_model(self):
        fields = {}
        for field_name, field_schema in self.fields_schema.items():
            field_type = self._map_type(field_schema['type'])
            fields[field_name] = (field_type, ...)
        return create_model('DynamicModel', **fields)

    def _map_type(self, milvus_type: str):
        type_mapping = {
            'INT64': int,
            'DOUBLE': float,
            'BOOL': bool,
            'VARCHAR': str,
            'FLOAT_VECTOR': list,
        }
        return type_mapping.get(milvus_type, Any)

    def validate(self, data: Dict[str, Any]):
        return self.pydantic_model(**data)

from pydantic import BaseModel, create_model, Field, validator
from typing import Dict, Any, List, Union, Optional
from enum import IntEnum


class MilvusDataType(IntEnum):
    NONE = 0
    BOOL = 1
    INT8 = 2
    INT16 = 3
    INT32 = 4
    INT64 = 5
    FLOAT = 10
    DOUBLE = 11
    STRING = 20
    VARCHAR = 21
    ARRAY = 22
    JSON = 23
    BINARY_VECTOR = 100
    FLOAT_VECTOR = 101
    FLOAT16_VECTOR = 102
    BFLOAT16_VECTOR = 103
    SPARSE_FLOAT_VECTOR = 104
    UNKNOWN = 999


class DatasetSchema:
    def __init__(self, fields_schema: Dict[str, Dict[str, Any]]):
        self.fields_schema = fields_schema
        self.pydantic_model = self._create_pydantic_model()

    def _create_pydantic_model(self):
        fields = {}
        for field_name, field_schema in self.fields_schema.items():
            field_type, field_constraints = self._map_type(field_schema)
            fields[field_name] = (field_type, Field(**field_constraints))

        validators = {}
        for field_name, field_schema in self.fields_schema.items():
            if field_schema['type'] == MilvusDataType.ARRAY:
                validators[f'validate_{field_name}'] = self._create_array_validator(field_schema)

        model = create_model('DynamicModel', **fields)
        for validator_name, validator_func in validators.items():
            setattr(model, validator_name, validator(field_name)(classmethod(validator_func)))

        return model

    def _map_type(self, field_schema: Dict[str, Any]) -> tuple:
        milvus_type = field_schema['type']
        if isinstance(milvus_type, str):
            milvus_type = MilvusDataType[milvus_type.upper()]
        elif isinstance(milvus_type, int):
            milvus_type = MilvusDataType(milvus_type)

        constraints = self._get_field_constraints(field_schema)

        type_mapping = {
            MilvusDataType.BOOL: (bool, constraints),
            MilvusDataType.INT8: (int, constraints),
            MilvusDataType.INT16: (int, constraints),
            MilvusDataType.INT32: (int, constraints),
            MilvusDataType.INT64: (int, constraints),
            MilvusDataType.FLOAT: (float, constraints),
            MilvusDataType.DOUBLE: (float, constraints),
            MilvusDataType.STRING: (str, constraints),
            MilvusDataType.VARCHAR: (str, constraints),
            MilvusDataType.JSON: (Dict[str, Any], constraints),
            MilvusDataType.ARRAY: (List[Any], constraints),
            MilvusDataType.BINARY_VECTOR: (List[int], constraints),
            MilvusDataType.FLOAT_VECTOR: (List[float], constraints),
            MilvusDataType.FLOAT16_VECTOR: (List[float], constraints),
            MilvusDataType.BFLOAT16_VECTOR: (List[float], constraints),
            MilvusDataType.SPARSE_FLOAT_VECTOR: (Dict[int, float], constraints),
        }
        return type_mapping.get(milvus_type, (Any, constraints))

    def _get_field_constraints(self, field_schema: Dict[str, Any]) -> Dict[str, Any]:
        constraints = {}
        if 'dim' in field_schema:
            constraints['min_items'] = field_schema['dim']
            constraints['max_items'] = field_schema['dim']
        if 'max_length' in field_schema:
            constraints['max_length'] = field_schema['max_length']
        return constraints

    def _create_array_validator(self, field_schema: Dict[str, Any]):
        element_type = field_schema.get('element_type', Any)
        element_max_length = field_schema.get('element_max_length')

        def validate_array(cls, v):
            if not isinstance(v, list):
                raise ValueError("Must be a list")

            for item in v:
                if not isinstance(item, self._map_type({'type': element_type})[0]):
                    raise ValueError(f"All elements must be of type {element_type}")

                if element_max_length and isinstance(item, str) and len(item) > element_max_length:
                    raise ValueError(f"String elements must not exceed {element_max_length} characters")

            return v

        return validate_array

    def validate(self, data: Dict[str, Any]):
        return self.pydantic_model(**data)


# Usage example
if __name__ == "__main__":
    schema = {
        "id": {"type": "INT64"},
        "name": {"type": "VARCHAR", "max_length": 100},
        "age": {"type": "INT32"},
        "score": {"type": "FLOAT"},
        "is_active": {"type": "BOOL"},
        "embedding": {"type": "FLOAT_VECTOR", "dim": 128},
        "tags": {"type": "ARRAY", "element_type": "VARCHAR", "element_max_length": 50},
        "metadata": {"type": "JSON"}
    }

    dataset_schema = DatasetSchema(schema)

    # Validate data
    valid_data = {
        "id": 1,
        "name": "John Doe",
        "age": "30",
        "score": 95.5,
        "is_active": True,
        "embedding": [0.1] * 128,
        "tags": ["tag1", "tag2"],
        "metadata": {"key": "value"}
    }

    validated_data = dataset_schema.validate(valid_data)
    print(validated_data)

    # Test invalid data
    try:
        invalid_data = valid_data.copy()
        invalid_data["tags"] = ["tag1", "this_is_a_very_long_tag_that_exceeds_the_max_length"*20]
        dataset_schema.validate(invalid_data)
    except ValueError as e:
        print(f"Validation error: {e}")

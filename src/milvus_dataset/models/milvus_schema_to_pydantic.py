from pydantic import BaseModel, create_model, Field, field_validator
from typing import Dict, Any, List, Union, Optional
from pymilvus import DataType, FieldSchema


class DatasetSchema:
    def __init__(self, fields: List[FieldSchema]):
        self.fields = fields
        self.pydantic_model = self._create_pydantic_model()

    def _create_pydantic_model(self):
        fields = {}
        for field in self.fields:
            field_type, field_constraints = self._map_type(field)
            field_name = "json_field" if field.name == "json" else field.name
            fields[field_name] = (field_type, Field(**field_constraints))

        validators = {}
        for field in self.fields:
            if field.dtype == DataType.ARRAY:
                validators[f'validate_{field.name}'] = self._create_array_validator(field)

        model = create_model('DynamicModel', **fields)
        for validator_name, validator_func in validators.items():
            setattr(model, validator_name, field_validator(field.name)(classmethod(validator_func)))

        return model

    def _map_type(self, field: FieldSchema) -> tuple:
        constraints = self._get_field_constraints(field)

        type_mapping = {
            DataType.BOOL: (bool, constraints),
            DataType.INT8: (int, constraints),
            DataType.INT16: (int, constraints),
            DataType.INT32: (int, constraints),
            DataType.INT64: (int, constraints),
            DataType.FLOAT: (float, constraints),
            DataType.DOUBLE: (float, constraints),
            DataType.STRING: (str, constraints),
            DataType.VARCHAR: (str, constraints),
            DataType.JSON: (Dict[str, Any], constraints),
            DataType.BINARY_VECTOR: (List[int], constraints),
            DataType.FLOAT_VECTOR: (List[float], constraints),
        }

        if field.dtype == DataType.ARRAY:
            element_type = self._map_type(FieldSchema('temp', field.element_type))[0]
            return (List[element_type], constraints)

        return type_mapping.get(field.dtype, (Any, constraints))

    def _get_field_constraints(self, field: FieldSchema) -> Dict[str, Any]:
        constraints = {}
        if field.dtype == DataType.FLOAT_VECTOR:
            constraints['min_items'] = field.dim
            constraints['max_items'] = field.dim
        if field.dtype == DataType.VARCHAR:
            constraints['max_length'] = field.max_length
        if field.dtype == DataType.ARRAY:
            constraints['max_items'] = field.max_capacity
        if not field.is_primary:
            constraints['default'] = None
        return constraints

    def _create_array_validator(self, field: FieldSchema):
        element_type = self._map_type(FieldSchema('temp', field.element_type))[0]
        max_capacity = field.max_capacity

        def validate_array(cls, v):
            if not isinstance(v, list):
                raise ValueError("Must be a list")

            if len(v) > max_capacity:
                raise ValueError(f"Array length must not exceed {max_capacity}")

            for item in v:
                if not isinstance(item, element_type):
                    raise ValueError(f"All elements must be of type {element_type}")

            return v

        return validate_array

    def validate(self, data: Dict[str, Any]):
        if "json" in data:
            data["json_field"] = data.pop("json")
        return self.pydantic_model(**data)


# Usage example
if __name__ == "__main__":
    default_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="float", dtype=DataType.FLOAT),
        FieldSchema(name="varchar", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="json", dtype=DataType.JSON),
        FieldSchema(name="int_array", dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=1024),
        FieldSchema(name="varchar_array", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=1024,
                    max_length=65535),
        FieldSchema(name="float_vector", dtype=DataType.FLOAT_VECTOR, dim=128)
    ]

    dataset_schema = DatasetSchema(default_fields)

    # Validate data
    valid_data = {
        "id": 1,
        "float": 95.5,
        "varchar": "John Doe",
        "json": {"key": "value"},
        "int_array": [1, 2, 3],
        "varchar_array": ["tag1", "tag2"],
        "float_vector": [0.1] * 128
    }

    validated_data = dataset_schema.validate(valid_data)
    print(validated_data)

    # Test invalid data
    try:
        invalid_data = valid_data.copy()
        invalid_data["int_array"] = [1, 2, "3"]  # Invalid type in array
        dataset_schema.validate(invalid_data)
    except ValueError as e:
        print(f"Validation error: {e}")

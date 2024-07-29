from pydantic import BaseModel, Field, create_model
from typing import List, Dict, Any, Union
from enum import IntEnum


class DataType(IntEnum):
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


class FieldSchema:
    def __init__(self, name, dtype, is_primary=False, max_length=None, element_type=None, max_capacity=None, dim=None):
        self.name = name
        self.dtype = dtype
        self.is_primary = is_primary
        self.max_length = max_length
        self.element_type = element_type
        self.max_capacity = max_capacity
        self.dim = dim


def create_field_model(field_schema: FieldSchema):
    field_type = Any
    field_kwargs = {}

    if field_schema.dtype == DataType.INT64:
        field_type = int
    elif field_schema.dtype == DataType.FLOAT:
        field_type = float
    elif field_schema.dtype == DataType.VARCHAR:
        field_type = str
        if field_schema.max_length:
            field_kwargs['max_length'] = field_schema.max_length
    elif field_schema.dtype == DataType.JSON:
        field_type = Dict[str, Any]
    elif field_schema.dtype == DataType.ARRAY:
        if field_schema.element_type == DataType.INT64:
            field_type = List[int]
        elif field_schema.element_type == DataType.VARCHAR:
            field_type = List[str]
        if field_schema.max_capacity:
            field_kwargs['max_items'] = field_schema.max_capacity
        if field_schema.max_length:
            field_kwargs['item_type'] = str
            field_kwargs['max_length'] = field_schema.max_length
    elif field_schema.dtype == DataType.FLOAT_VECTOR:
        field_type = List[float]
        if field_schema.dim:
            field_kwargs['min_items'] = field_schema.dim
            field_kwargs['max_items'] = field_schema.dim

    return (field_type, Field(..., **field_kwargs))


def create_schema_model(field_schemas: List[FieldSchema]):
    fields = {}
    for schema in field_schemas:
        fields[schema.name] = create_field_model(schema)

    return create_model('DynamicSchemaModel', **fields)


# 示例使用
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

DynamicModel = create_schema_model(default_fields)
print(DynamicModel)
# 使用动态生成的模型创建实例
instance = DynamicModel(
    id="1",
    float=3.14,
    varchar="Hello, world!",
    json={"key": "value"},
    int_array=[1, "2", 3],
    varchar_array=["a", "b", "c"],
    float_vector=[0.1] * 128
)

print(instance)

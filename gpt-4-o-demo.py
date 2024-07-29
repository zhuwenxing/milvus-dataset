from pydantic import BaseModel, Field, create_model, conlist, constr, ValidationError, field_validator
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


def get_base_type(data_type: DataType):
    if data_type in [DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64]:
        return int
    elif data_type in [DataType.FLOAT, DataType.DOUBLE]:
        return float
    elif data_type in [DataType.STRING, DataType.VARCHAR]:
        return str
    elif data_type == DataType.BOOL:
        return bool
    elif data_type == DataType.JSON:
        return Dict[str, Any]
    else:
        return Any


def create_field_model(field_schema: FieldSchema):
    field_type = get_base_type(field_schema.dtype)
    field_kwargs = {}

    if field_schema.dtype == DataType.VARCHAR and field_schema.max_length:
        field_type = constr(max_length=field_schema.max_length)

    elif field_schema.dtype == DataType.ARRAY:
        element_type = get_base_type(field_schema.element_type)
        if field_schema.max_capacity:
            field_type = conlist(element_type, max_length=field_schema.max_capacity)
        else:
            field_type = List[element_type]

        if field_schema.element_type == DataType.VARCHAR and field_schema.max_length:
            field_type = conlist(constr(max_length=field_schema.max_length), max_length=field_schema.max_capacity)

    elif field_schema.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
        if field_schema.dim:
            field_type = conlist(float if field_schema.dtype == DataType.FLOAT_VECTOR else int,
                                 min_length=field_schema.dim, max_length=field_schema.dim)

    return (field_type, Field(..., **field_kwargs))


def create_schema_model(field_schemas: List[FieldSchema]):
    fields = {}
    validators = {}
    for schema in field_schemas:
        fields[schema.name] = create_field_model(schema)
        if schema.dtype == DataType.ARRAY:
            validators[f'validate_{schema.name}'] = field_validator(schema.name)(
                create_array_validator(schema.element_type))

    return create_model('DynamicSchemaModel', **fields, __validators__=validators)


def create_array_validator(element_type: DataType):
    base_type = get_base_type(element_type)

    def validate_array(cls, v):
        for item in v:
            if not isinstance(item, base_type):
                raise ValueError(
                    f"All items must be of type {base_type.__name__}. Found item of type {type(item).__name__}")
        return v

    return validate_array


# 创建测试模型
test_fields = [
    FieldSchema(name="bool_field", dtype=DataType.BOOL),
    FieldSchema(name="int8_field", dtype=DataType.INT8),
    FieldSchema(name="int16_field", dtype=DataType.INT16),
    FieldSchema(name="int32_field", dtype=DataType.INT32),
    FieldSchema(name="int64_field", dtype=DataType.INT64),
    FieldSchema(name="float_field", dtype=DataType.FLOAT),
    FieldSchema(name="double_field", dtype=DataType.DOUBLE),
    FieldSchema(name="string_field", dtype=DataType.STRING),
    FieldSchema(name="varchar_field", dtype=DataType.VARCHAR, max_length=10),
    FieldSchema(name="json_field", dtype=DataType.JSON),
    FieldSchema(name="int_array", dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=5),
    FieldSchema(name="varchar_array", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=3,
                max_length=5),
    FieldSchema(name="float_vector", dtype=DataType.FLOAT_VECTOR, dim=3),
    FieldSchema(name="binary_vector", dtype=DataType.BINARY_VECTOR, dim=3)
]

TestModel = create_schema_model(test_fields)


def test_model(data, expected_valid):
    try:
        instance = TestModel(**data)
        print(f"Valid: {expected_valid}")
        print(instance)
    except ValidationError as e:
        print(f"Valid: {not expected_valid}")
        print(e)
    print()


# 测试用例
test_cases = [
    ({
         "bool_field": True,
         "int8_field": 127,
         "int16_field": 32767,
         "int32_field": 2147483647,
         "int64_field": 9223372036854775807,
         "float_field": 3.14,
         "double_field": 3.14159265359,
         "string_field": "Hello",
         "varchar_field": "HelloWorld",
         "json_field": {"key": "value"},
         "int_array": [1, 2, 3, 4, 5],
         "varchar_array": ["a", "bb", "ccc"],
         "float_vector": [1.0, 2.0, 3.0],
         "binary_vector": [0, 1, 1]
     }, True),
    ({
         "bool_field": "True",
         "int8_field": 128,
         "int16_field": "32767",
         "int32_field": 2147483648,
         "int64_field": 9223372036854775808,
         "float_field": "3.14",
         "double_field": "3.14159265359",
         "string_field": 12345,
         "varchar_field": "HelloWorldTooLong",
         "json_field": "Not a JSON",
         "int_array": [1, 2, "3", 4, 5],
         "varchar_array": ["a", "bb", "cccccc"],
         "float_vector": [1.0, 2.0, 3.0, 4.0],
         "binary_vector": [0, 1, 2]
     }, False)
]

for case, expected_valid in test_cases:
    test_model(case, expected_valid)

# 额外测试 int_array 的严格验证
extra_tests = [
    {
        "bool_field": True,
        "int8_field": 127,
        "int16_field": 32767,
        "int32_field": 2147483647,
        "int64_field": 9223372036854775807,
        "float_field": 3.14,
        "double_field": 3.14159265359,
        "string_field": "Hello",
        "varchar_field": "HelloWorld",
        "json_field": {"key": "value"},
        "int_array": [1, 2, "3", 4, 5],  # 包含字符串 "3"
        "varchar_array": ["a", "bb", "ccc"],
        "float_vector": [1.0, 2.0, 3.0],
        "binary_vector": [0, 1, 1]
    },
    {
        "bool_field": True,
        "int8_field": 127,
        "int16_field": 32767,
        "int32_field": 2147483647,
        "int64_field": 9223372036854775807,
        "float_field": 3.14,
        "double_field": 3.14159265359,
        "string_field": "Hello",
        "varchar_field": "HelloWorld",
        "json_field": {"key": "value"},
        "int_array": [1, 2, "a", 4, 5],  # 包含字符串 "a"
        "varchar_array": ["a", "bb", "ccc"],
        "float_vector": [1.0, 2.0, 3.0],
        "binary_vector": [0, 1, 1]
    }
]

for extra_test in extra_tests:
    test_model(extra_test, False)


class IntModel(BaseModel):
    int_field: int

hello = IntModel(int_field='1')
print(hello)

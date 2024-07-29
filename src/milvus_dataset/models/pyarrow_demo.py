import pyarrow as pa
import pandas as pd
from pymilvus import DataType, FieldSchema
from typing import List
import json


def milvus_to_pyarrow_type(field: FieldSchema):
    type_mapping = {
        DataType.BOOL: pa.bool_(),
        DataType.INT8: pa.int8(),
        DataType.INT16: pa.int16(),
        DataType.INT32: pa.int32(),
        DataType.INT64: pa.int64(),
        DataType.FLOAT: pa.float32(),
        DataType.DOUBLE: pa.float64(),
        DataType.STRING: pa.string(),
        DataType.VARCHAR: pa.string(),
        DataType.JSON: pa.struct([]),  # Use struct for JSON
        DataType.BINARY_VECTOR: pa.list_(pa.int8()),
        DataType.FLOAT_VECTOR: pa.list_(pa.float32()),
    }

    if field.dtype == DataType.ARRAY:
        element_type = milvus_to_pyarrow_type(FieldSchema('temp', field.element_type))
        return pa.list_(element_type)

    return type_mapping.get(field.dtype, pa.string())  # Default to string for unknown types


def create_pyarrow_schema(fields: List[FieldSchema]):
    pa_fields = []
    for field in fields:
        pa_type = milvus_to_pyarrow_type(field)
        pa_fields.append(pa.field(field.name, pa_type))
    return pa.schema(pa_fields)


def json_to_pyarrow(json_str):
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError:
        return None


def validate_dataframe(df: pd.DataFrame, pa_schema: pa.Schema):
    # Pre-process JSON fields
    for field in pa_schema:
        if isinstance(field.type, pa.StructType):
            df[field.name] = df[field.name].apply(json_to_pyarrow)

    try:
        table = pa.Table.from_pandas(df, schema=pa_schema)
        return table.to_pandas(), None
    except pa.lib.ArrowInvalid as e:
        return None, str(e)


# 使用示例
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

    pa_schema = create_pyarrow_schema(default_fields)
    print("PyArrow Schema:")
    print(pa_schema)

    # 创建有效的 DataFrame
    valid_data = pd.DataFrame({
        "id": [1, 2],
        "float": [95.5, 88.3],
        "varchar": ["John Doe", "Jane Smith"],
        "json": {"a":1},
        "int_array": [[1, 2, 3], [4, 5, 6]],
        "varchar_array": [["tag1", "tag2"], ["tag3", "tag4"]],
        "float_vector": [[0.1] * 128, [0.2] * 128]
    })

    validated_df, error = validate_dataframe(valid_data, pa_schema)
    if error:
        print(f"Validation error: {error}")
    else:
        print("Validated DataFrame:")
        print(validated_df)

    # 测试无效数据
    invalid_data = valid_data.copy()
    invalid_data.at[0, "json"] = "not a valid json"

    validated_df, error = validate_dataframe(invalid_data, pa_schema)
    if error:
        print(f"Validation error: {error}")
    else:
        print("Validated DataFrame:")
        print(validated_df)

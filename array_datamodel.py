from pydantic import BaseModel, Field, create_model
from typing import List
from enum import Enum


class DataType(str, Enum):
    ARRAY = "array"
    VARCHAR = "varchar"


class FieldSchema(BaseModel):
    name: str
    dtype: DataType
    element_type: DataType
    max_capacity: int
    max_length: int


# 创建动态模型
def create_dynamic_model(field_schema: FieldSchema):
    field_type = List[str]
    field_constraints = {
        "max_items": field_schema.max_capacity,
        "max_length": field_schema.max_length
    }

    return create_model(
        f"DynamicModel_{field_schema.name}",
        **{field_schema.name: (field_type, Field(**field_constraints))}
    )


# 使用示例
field_schema = FieldSchema(
    name="varchar_array",
    dtype=DataType.ARRAY,
    element_type=DataType.VARCHAR,
    max_capacity=10,
    max_length=10
)

DynamicModel = create_dynamic_model(field_schema)

# 测试模型
instance = DynamicModel(varchar_array=["test1", "test2"])
print(instance)

# 验证
try:
    DynamicModel(varchar_array=["a" * 65536])  # 应该引发错误
except Exception as e:
    print(f"Validation length error: {e}")

try:
    DynamicModel(varchar_array=["test" for _ in range(1025)])  # 应该引发错误
except Exception as e:
    print(f"Validation cap error: {e}")

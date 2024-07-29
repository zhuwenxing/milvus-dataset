import pyarrow as pa

EMBEDDING_SIZE = 128

schema = pa.schema([
    ('id', pa.int64()),
    ('text', pa.string()),
    ('embedding', pa.list_(pa.float32(), EMBEDDING_SIZE))
])


data = [
    {'id': 1, 'text': 'Hello', 'embedding': [0.1] * EMBEDDING_SIZE},
    {'id': 2, 'text': 'World', 'embedding': [0.2] * (EMBEDDING_SIZE+1)},
]


table = pa.Table.from_pylist(data, schema=schema)
print(table)


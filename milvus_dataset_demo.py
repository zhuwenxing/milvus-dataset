from milvus_dataset import list_datasets, load_dataset, create_dataset, ConfigManager

config_manager = ConfigManager()
config_manager.init_local_storage("/tmp/milvus_dataset")

print(config_manager)
d_list = list_datasets()
print(d_list)

dataset = create_dataset("test_dataset")
dataset.write(
    {
        "id": range(100),
        "text": [f"text_{i}" for i in range(100)],
        "value": [i * 0.1 for i in range(100)],
    }
)

demo = load_dataset("test_dataset")
print(demo)
for d in demo:
    print(d)

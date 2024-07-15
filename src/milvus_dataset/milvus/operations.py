from pymilvus import Collection, connections


class MilvusOperations:
    def __init__(self, dataset):
        self.dataset = dataset

    def to_milvus(self, mode: str = 'bulk'):
        connections.connect(**self.dataset.metadata['milvus_connection'])
        collection = Collection(self.dataset.name)

        if mode == 'bulk':
            data = self.dataset.read(mode='full')
            collection.insert(data.to_dict('records'))
        elif mode == 'stream':
            for batch in self.dataset.read(mode='batch', batch_size=1000):
                collection.insert(batch.to_dict('records'))
        else:
            raise ValueError("Invalid mode. Expected 'bulk' or 'stream'.")

        collection.flush()
        print(f"Data imported to Milvus collection: {self.dataset.name}")

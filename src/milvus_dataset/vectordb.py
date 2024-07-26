import numpy as np
import time
import statistics
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

class VectorDBManager(ABC):
    def __init__(self, host, port, collection_name, dim):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self.collection = None
        self.benchmark_results = Manager().dict()

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def create_collection(self):
        pass

    @abstractmethod
    def insert(self, vectors):
        pass

    @abstractmethod
    def bulk_import(self, file_path):
        pass
    @abstractmethod
    def search(self, query_vectors, top_k=10):
        pass

    @abstractmethod
    def count(self):
        pass

    def generate_random_vectors(self, num_vectors):
        return np.random.rand(num_vectors, self.dim).astype(np.float32)

    def benchmark_insert(self, num_vectors, batch_size=1000, num_rounds=5, num_processes=None):
        if num_processes is None:
            num_processes = cpu_count()

        results = []
        for _ in range(num_rounds):
            vectors = self.generate_random_vectors(num_vectors)
            batches = [vectors[i:i + batch_size] for i in range(0, num_vectors, batch_size)]

            start_time = time.perf_counter()

            with Pool(processes=num_processes) as pool:
                pool.map(self.insert, batches)

            end_time = time.perf_counter()
            total_time = end_time - start_time
            results.append(total_time)

        return {
            "operation": "insert",
            "num_vectors": num_vectors,
            "batch_size": batch_size,
            "num_rounds": num_rounds,
            "num_processes": num_processes,
            "avg_time": statistics.mean(results),
            "min_time": min(results),
            "max_time": max(results),
            "std_dev": statistics.stdev(results) if len(results) > 1 else 0,
            "avg_throughput": num_vectors / statistics.mean(results)
        }

    def benchmark_search(self, num_queries, top_k=10, num_rounds=5, num_processes=None):
        if num_processes is None:
            num_processes = cpu_count()

        results = []
        for _ in range(num_rounds):
            query_vectors = self.generate_random_vectors(num_queries)

            start_time = time.perf_counter()

            with Pool(processes=num_processes) as pool:
                search_func = partial(self.search, top_k=top_k)
                pool.map(search_func, [query_vectors[i:i + 1] for i in range(num_queries)])

            end_time = time.perf_counter()
            total_time = end_time - start_time
            results.append(total_time)

        return {
            "operation": "search",
            "num_queries": num_queries,
            "top_k": top_k,
            "num_rounds": num_rounds,
            "num_processes": num_processes,
            "avg_time": statistics.mean(results),
            "min_time": min(results),
            "max_time": max(results),
            "std_dev": statistics.stdev(results) if len(results) > 1 else 0,
            "avg_qps": num_queries / statistics.mean(results)
        }

    def benchmark_mixed_workload(self, num_insert, num_search, insert_batch_size=1000, search_top_k=10, duration=60,
                                 num_processes=None):
        if num_processes is None:
            num_processes = cpu_count()

        start_time = time.perf_counter()
        end_time = start_time + duration
        insert_count = 0
        search_count = 0

        with Pool(processes=num_processes) as pool:
            while time.perf_counter() < end_time:
                # Perform inserts
                insert_vectors = self.generate_random_vectors(num_insert)
                pool.apply_async(self.insert, (insert_vectors,))
                insert_count += num_insert

                # Perform searches
                query_vectors = self.generate_random_vectors(num_search)
                pool.apply_async(self.search, (query_vectors, search_top_k))
                search_count += num_search

        total_time = time.perf_counter() - start_time
        return {
            "operation": "mixed_workload",
            "duration": total_time,
            "insert_count": insert_count,
            "search_count": search_count,
            "num_processes": num_processes,
            "insert_throughput": insert_count / total_time,
            "search_throughput": search_count / total_time
        }

    def run_comprehensive_benchmark(self, insert_count=100000, search_count=10000, top_k=10, num_processes=None,
                                    mixed_duration=300):
        if num_processes is None:
            num_processes = cpu_count()

        print("Starting comprehensive benchmark...")

        print("Benchmarking insert...")
        insert_results = self.benchmark_insert(insert_count, num_processes=num_processes)
        self.benchmark_results["insert"] = insert_results
        print(f"Insert results: {insert_results}")

        print("Benchmarking search...")
        search_results = self.benchmark_search(search_count, top_k, num_processes=num_processes)
        self.benchmark_results["search"] = search_results
        print(f"Search results: {search_results}")

        print("Benchmarking mixed workload...")
        mixed_results = self.benchmark_mixed_workload(insert_count // 10, search_count // 10, duration=mixed_duration,
                                                      num_processes=num_processes)
        self.benchmark_results["mixed_workload"] = mixed_results
        print(f"Mixed workload results: {mixed_results}")

        print("Benchmark completed.")

        return dict(self.benchmark_results)


class MilvusVectorManager(VectorDBManager):
    def connect(self):
        connections.connect("default", host=self.host, port=self.port)

    def create_collection(self):
        if self.collection is None:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]
            schema = CollectionSchema(fields, "Vector collection for Milvus")
            self.collection = Collection(self.collection_name, schema)

    def insert(self, vectors):
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)

        entities = [
            {"embedding": vec.tolist()} for vec in vectors
        ]
        self.collection.insert(entities)
        self.collection.flush()

    def bulk_import(self, file_path):
        pass

    def search(self, query_vectors, top_k=10):
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            query_vectors, "embedding", search_params, top_k=top_k, output_fields=["id"]
        )
        return results

    def count(self):
        return self.collection.num_entities


if __name__ == "__main__":
    milvus_manager = MilvusVectorManager("localhost", 19530, "test_collection", 128)
    milvus_manager.connect()
    milvus_manager.create_collection()

    benchmark_results = milvus_manager.run_comprehensive_benchmark(insert_count=100000, search_count=10000)
    print(benchmark_results)

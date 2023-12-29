from pymilvus import connections
from pymilvus import utility

# 1.连接milvus
connections.connect(
    alias="default",
    user="root",
    password="Milvusmdpi",
    host="localhost",
    port="19530",
)

# 2.创建collection
from pymilvus import CollectionSchema, FieldSchema, DataType

book_id = FieldSchema(
    name="book_id",
    dtype=DataType.INT64,
    is_primary=True,
)
book_name = FieldSchema(
    name="book_name",
    dtype=DataType.VARCHAR,
    max_length=200,
)
word_count = FieldSchema(
    name="word_count",
    dtype=DataType.INT64,
)
book_intro = FieldSchema(name="book_intro", dtype=DataType.FLOAT_VECTOR, dim=2)
schema = CollectionSchema(
    fields=[book_id, book_name, word_count, book_intro], description="Test book search"
)
collection_name = "book"

from pymilvus import Collection

if not utility.has_collection(collection_name):
    collection = Collection(
        name=collection_name, schema=schema, using="default", shards_num=2
    )
else:
    utility.drop_collection(collection_name)
    collection = Collection(
        name=collection_name, schema=schema, using="default", shards_num=2
    )

# 3.检查collection是否存在


print(utility.has_collection("book"))
print(collection.schema)
print(collection.description)
print(collection.indexes)

# 4.创建partition
if not collection.has_partition("novel"):
    collection.create_partition("novel")
else:
    print(collection.has_partition("novel"))
    print(collection.partitions)

# 5.插入数据

import random

data = [
    [i for i in range(2000)],
    [str(i) for i in range(2000)],
    [i for i in range(10000, 12000)],
    [[random.random() for _ in range(2)] for _ in range(2000)],
]

collection.insert(data, partition_name="novel")

# 6.删除数据
expr = "book_id in [0,1]"
collection.delete(expr)

# 7.构建索引
collection = Collection("book")
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024},
}
collection.release()
collection.create_index(field_name="book_intro", index_params=index_params)
utility.index_building_progress("book")


# 8.搜索
collection = Collection("book")
collection.load()
search_params = {"metric_type": "L2", "params": {"nprobe": 10}, "offset": 5}

results = collection.search(
    data=[[0.1, 0.2]],
    anns_field="book_intro",
    param=search_params,
    limit=10,
    expr=None,
    output_fields=[
        "book_id",
        "book_name",
    ],  # set the names of the fields you want to retrieve from the search result.
    consistency_level="Strong",
)

print(results[0].ids)

print(results[0].distances)

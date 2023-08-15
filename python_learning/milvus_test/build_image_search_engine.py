import csv
from glob import glob
from pathlib import Path
from statistics import mean

from towhee.dc2 import pipe, ops, DataCollection
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from matplotlib import pyplot as plt

# Towhee parameters
MODEL = "resnet50"
DEVICE = None  # if None, use default device (cuda is enabled if available)

# Milvus parameters
HOST = "localhost"
PORT = "19530"
user = "root"
password = "Milvusmdpi"
TOPK = 10
DIM = 2048  # dimension of embedding extracted by MODEL
COLLECTION_NAME = "reverse_image_search"
INDEX_TYPE = "IVF_FLAT"
METRIC_TYPE = "L2"

# path to csv (column_1 indicates image path) OR a pattern of image paths
INSERT_SRC = "/home/bocheng/data/images/reverse_image_search/reverse_image_search.csv"
QUERY_SRC = "/home/bocheng/data/images/reverse_image_search/test/*/*.JPEG"


# Load image path
def load_image(x):
    if x.endswith("csv"):
        with open(x) as f:
            reader = csv.reader(f)
            next(reader)
            for item in reader:
                yield item[1]
    else:
        for item in glob(x):
            yield item


# Embedding pipeline
p_embed = (
    pipe.input("src")
    .flat_map("src", "img_path", load_image)
    .map("img_path", "img", ops.image_decode())
    .map("img", "vec", ops.image_embedding.timm(model_name=MODEL, device=DEVICE))
)

# Display embedding result, no need for implementation
p_display = p_embed.output("img_path", "img", "vec")
# out=DataCollection(p_display('/home/bocheng/data/images/reverse_image_search/test/goldfish/*.JPEG'))[0]
# print(out["img_path"])
# # DataCollection(p_display('/home/bocheng/data/images/reverse_image_search/test/goldfish/*.JPEG')).show()
# plt.imshow(out["img"])
# plt.show()


# Create milvus collection (delete first if exists)
def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(
            name="path",
            dtype=DataType.VARCHAR,
            description="path to image",
            max_length=500,
            is_primary=True,
            auto_id=False,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            description="image embedding vectors",
            dim=dim,
        ),
    ]
    schema = CollectionSchema(fields=fields, description="reverse image search")
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        "metric_type": METRIC_TYPE,
        "index_type": INDEX_TYPE,
        "params": {"nlist": 2048},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


# Connect to Milvus service
connections.connect(host=HOST, port=PORT, user=user, password=password)

# Create collection
collection = create_milvus_collection(COLLECTION_NAME, DIM)
print(f"A new collection created: {COLLECTION_NAME}")

# # Insert pipeline
# p_insert = (
#         p_embed.map(('img_path', 'vec'), 'mr', ops.ann_insert.milvus_client(
#                     host=HOST,
#                     port=PORT,
#                     user=user,
#                     password=password,
#                     collection_name=COLLECTION_NAME
#                     ))
#           .output('mr')
# )

# # Insert data
# p_insert(INSERT_SRC)

# # Check collection
# print('Number of data inserted:', collection.num_entities)

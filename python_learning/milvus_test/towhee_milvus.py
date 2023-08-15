import time
from towhee import pipeline
import os
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from typing import List

image_type = [".jpeg", ".jpg", ".png"]


class ImageEmbedding:
    def __init__(
        self,
        host,
        port,
        user,
        password,
        index_type="IVF_FLAT",
        metric_type="L2",
        partition_name="image_search_test",
        collection_name="reverse_image_search",
    ):
        connections.connect(host=host, port=port, user=user, password=password)
        print("milvus connection established")
        self.collection_name = collection_name
        self.index_type = index_type
        self.metric_type = metric_type
        self.partition_name = partition_name
        self.img_pipeline = pipeline(
            "towhee/image-embedding-vitlarge"
        )  # {"image-embedding":"ResNet50"} "towhee/image-embedding-swinbase"

    def create_collection(self) -> Collection:
        if not utility.has_collection(self.collection_name):
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
                    dim=1024,
                ),
            ]
            schema = CollectionSchema(fields=fields, description="reverse image search")
            collection = Collection(name=self.collection_name, schema=schema)

            index_params = {
                "metric_type": self.metric_type,
                "index_type": self.index_type,
                "params": {"nlist": 2048},
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            print(
                f'collection "{self.collection_name}" has been created successfully !'
            )
        else:
            collection = Collection(self.collection_name)
            print(f"collection {self.collection_name} has been loaded successfully !")
        self.create_partition(collection)
        return collection

    def image_embedding(self, image_path: str):
        return self.img_pipeline(image_path)

    def images_embeddings(self, image_dir: str, num: int = None):
        start_time = time.time()
        embedding_list = []
        images_paths = self.filter_images(image_dir)
        if num is not None:
            images_paths = images_paths[: min(num, len(images_paths))]
        for image_path in images_paths:
            embedding = self.image_embedding(image_path)
            embedding_list.append(embedding)
        end_time = time.time()
        print(
            f"image to embedding finished,total {len(embedding_list)} images,total time consumed: {(end_time-start_time)}"
        )
        return images_paths, embedding_list

    def create_partition(self, collection: Collection):
        if not collection.has_partition(self.partition_name):
            collection.create_partition(self.partition_name)
            print(f'partition "{self.partition_name}" has been created !')

    @staticmethod
    def filter_images(image_dir: str):
        filtered_images_paths = []
        for file in os.walk(image_dir):
            for f in file[2]:
                image_path = os.path.join(file[0], f)
                suffix = os.path.splitext(image_path)[-1].lower()
                if suffix in image_type:
                    filtered_images_paths.append(image_path)
        return filtered_images_paths

    def insert_images(
        self,
        collection: Collection,
        images_paths: List[str],
        embedding_list: List[List],
    ):
        collection.insert(
            data=[images_paths, embedding_list], partition_name=self.partition_name
        )
        print(f"total {len(images_paths)} images inserted")

    def query_topk(self, collection: Collection, embedding_one: List, topk: int = 10):
        collection.load()
        search_params = {"metric_type": self.metric_type}
        res = collection.search(
            data=[embedding_one],
            anns_field="embedding",
            limit=topk,
            param=search_params,
        )
        return res


def drop_collection(collection_name: str):
    utility.drop_collection(collection_name)


if __name__ == "__main__":
    start_time = time.time()
    image_dir = "/home/bocheng/data/images/articles/biomedicines-1324322"
    image_path = os.path.join(image_dir, "figure_1.jpg")
    absolute_path = os.path.abspath(image_path)
    # # print(embedding.shape)

    HOST = "localhost"
    PORT = "19530"
    user = "root"
    password = "Milvusmdpi"
    TOPK = 10
    DIM = 2048  # dimension of embedding extracted by MODEL
    COLLECTION_NAME = "vitlarge"
    INDEX_TYPE = "IVF_FLAT"
    METRIC_TYPE = "L2"
    partition_name = "vitlarge"

    imageEmbedding = ImageEmbedding(
        HOST,
        PORT,
        user,
        password,
        INDEX_TYPE,
        METRIC_TYPE,
        partition_name,
        COLLECTION_NAME,
    )
    collection = imageEmbedding.create_collection()

    image_embedding = imageEmbedding.image_embedding(image_path)
    # imageEmbedding.insert_images(collection, [absolute_path], [image_embedding])
    res = imageEmbedding.query_topk(collection, image_embedding, 6)
    print(res)

    # # connections.connect(user=user, password=password, host=HOST, port=PORT)
    # # collection = Collection("reverse_image_search")
    # # # print(collection.partition)
    # # collection.insert(data=[[absolute_path], [image_embedding]])

    # image_dir = "/home/bocheng/data/images/articles/biomedicines-1324322"
    # images_paths, embedding_list = imageEmbedding.images_embeddings(image_dir)
    # imageEmbedding.insert_images(collection, images_paths, embedding_list)

    end_time = time.time()
    print(f"time consumed is {end_time-start_time}秒")

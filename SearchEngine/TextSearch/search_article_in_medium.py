# -*-coding:utf-8 -*-

"""
# File       : search_article_in_medium.py
# Time       ：2023/5/24 9:45
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import pandas as pd
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

filepath = '/home/bocheng/data/corpus/New_Medium_Data.csv'


# 1.加载数据
def read_data(filepath: str):
	df = pd.read_csv(filepath, converters={'title_vector': lambda x: eval(x)})


# 2.创建collection
connections.connect(host='127.0.0.1', port='19530')


def create_milvus_collection(collection_name, dim):
	if utility.has_collection(collection_name):
		collection = Collection(collection_name)
	else:
		fields = [
			FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
			FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
			FieldSchema(name="title_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
			FieldSchema(name="link", dtype=DataType.VARCHAR, max_length=500),
			FieldSchema(name="reading_time", dtype=DataType.INT64),
			FieldSchema(name="publication", dtype=DataType.VARCHAR, max_length=500),
			FieldSchema(name="claps", dtype=DataType.INT64),
			FieldSchema(name="responses", dtype=DataType.INT64)
		]
		schema = CollectionSchema(fields=fields, description='search text')
		collection = Collection(name=collection_name, schema=schema)

		index_params = {
			'metric_type': "L2",
			'index_type': "IVF_FLAT",
			'params': {"nlist": 2048}
		}
		collection.create_index(field_name='title_vector', index_params=index_params)
	return collection


collection = create_milvus_collection('search_article_in_medium', 768)
from pymilvus import utility

# 3.向量化数据并插入milvus
from towhee.dc2 import ops

vec = ops.text_embedding.transformers(model_name='bert-base-cased')


def to_vec(x):
	return vec(x.title)[0]


def insert_data(collection: Collection, df):
	df['title_vector'] = df.apply(to_vec, axis=1)
	id = df.id.values.tolist()
	title = df.title.values.tolist()
	title_vector = list(map(lambda arr: arr.tolist(), df.title_vector.values))
	link = df.link.values.tolist()
	reading_time = df.reading_time.values.tolist()
	publication = df.publication.values.tolist()
	claps = df.claps.values.tolist()
	responses = df.responses.values.tolist()
	collection.insert(data=[id, title, title_vector, link, reading_time, publication, claps, responses])
	print(f"{len(id)} items has been inserted into {collection.name}")


# 4.向量检索
def query(collection: Collection, input_str: str, topk=10):
	embedding = vec(input_str)
	res = collection.search(data=embedding, anns_field='title_vector', param={"metric_type": 'L2'}, limit=topk,
							output_fields=['title'])
	return res


if __name__ == '__main__':
	# insert_data(collection,df)
	# collection.load()
	# print(collection.num_entities)
	res = query(collection, input_str='Adventure into Machine Learning using Python', topk=10)
	for r in res[0]:
		print(r.entity.value_of_field('title'))

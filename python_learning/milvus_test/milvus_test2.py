from pymilvus import utility, connections

connections.connect(
    alias="default",
    user="root",
    password="Milvusmdpi",
    host="localhost",
    port="19530",
)

collections = utility.list_collections()
print(collections)

# for collection in collections:
#     if  utility.has_collection(collection):
#         utility.drop_collection(collection_name=collection)
#         print(f"{collection} has been deleted:", not utility.has_collection(collection))
# utility.drop_collection(collection_name="test_collection")

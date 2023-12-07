from fastapi import FastAPI
from enum import Enum


class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


myapp = FastAPI()


@myapp.get("/")
async def root():
    return {"message": "Hello World"}


@myapp.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


@myapp.get("/user/me")
def read_user_me():
    return {"user_id": "the current user"}


@myapp.get("/user/{user_id}")
def read_user(user_id: str):
    return {"user_id": user_id}


@myapp.get("/model/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": "alexnet", "message": "Deep Learning FTW!"}
    # elif model_name is ModelName.lenet:
    #     return {"model_name": "lenet", "message": "LeCNN all the images"}
    # elif model_name is ModelName.resnet:
    #     return {"model_name": "resnet", "message": "Reservoir Learning"}
    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}


@myapp.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}

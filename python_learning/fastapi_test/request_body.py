from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI


class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None


app = FastAPI()


# @app.post("/items/")
# async def create_item(item: Item):
#     return item


@app.post("/items/{item_id}")
async def create_item(item_id: int, item: Item, q: Union[str, None] = None):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})

    item_dict.update({"item_id": item_id})
    if q:
        item_dict.update({"q": q})
    return item_dict

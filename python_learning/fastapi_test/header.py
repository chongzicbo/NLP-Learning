from typing import Union

from fastapi import FastAPI, Header, Request
from typing_extensions import Annotated

app = FastAPI()


@app.get("/items/")
async def read_items(
    content_type: Annotated[Union[str, None], Header()] = None,
):
    return {"Content-Type": content_type}


@app.get("/get_headers")
async def get_headers(request: Request):
    all_headers = request.headers
    for key, value in all_headers.items():
        print(key, value)
        # 例如，提取User-Agent头部的值
        if key == "User-Agent":
            user_agent = value
            print(f"User-Agent: {user_agent}")

        # 或者，提取某个特定Header的值
        if key == "Content-Type":
            content_type = value
            print(f"Content-Type: {content_type}")

    return {"message": "Headers received and printed."}

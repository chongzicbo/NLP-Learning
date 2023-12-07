from fastapi import FastAPI, Response, Cookie
from typing import Optional

app = FastAPI()


@app.get("/setcookie")
def get_setcookie(response: Response):
    response.set_cookie("cookie1", "cookie111111111111")
    response.set_cookie("cookie2", "cookie222222222222")
    response.set_cookie("cookie3", "cookie333333333333")
    return {"result": "设置cookie成功"}


@app.get("/getcookie")
def read_cookie(
    cookie1: Optional[str] = Cookie(None), cookie2: Optional[str] = Cookie(None)
):
    return {"cookie1": cookie1, "cookie2": cookie2}

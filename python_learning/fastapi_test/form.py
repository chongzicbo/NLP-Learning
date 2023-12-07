from fastapi import Form, FastAPI

app = FastAPI()


@app.post("/login")
async def login(username=Form(), password: str = Form()):
    return {"username": username}

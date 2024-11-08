from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
from loguru import logger

app = FastAPI()


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    upload_dir = "/tmp/uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir, exist_ok=True)
    logger.info(f"file will be uploaded to {upload_dir}")
    allowed_extensions = ["docx", "tex"]
    if not file.filename.lower().endswith(tuple(allowed_extensions)):
        raise HTTPException(status_code=400, detail="Invalid file type")
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    return JSONResponse(
        content={"filename": file.filename, "message": "File uploaded successfully"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=18001)

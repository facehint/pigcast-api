from fastapi import FastAPI, File, UploadFile, Response
from io import BytesIO

from starlette.responses import Response

from bighead.bighead import crop

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


def process_video():
    return "ok"


@app.post("/bighead/crop")
async def big_head_crop(file: UploadFile)\
        -> Response:
    contents = await file.read()
    result_bytes = crop(contents)
    return Response(content=result_bytes, media_type="image/png")
    # try:
    #     async with aiofiles.tempfile.NamedTemporaryFile("wb", delete=False) as temp:
    #         try:
    #             contents = await file.read()
    #             await temp.write(contents)
    #         except Exception:
    #             raise HTTPException(status_code=500, detail='Something went wrong')
    #         finally:
    #             await file.close()
    #
    #     res = await run_in_threadpool(process_video, temp.name)  # Pass temp.name to VideoCapture()
    # except Exception:
    #     raise HTTPException(status_code=500, detail='Something went wrong when processing the file')
    # finally:
    #     os.remove(temp.name)
    #
    # return res

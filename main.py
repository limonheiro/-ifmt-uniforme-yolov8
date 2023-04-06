from typing import List, BinaryIO
from fastapi import FastAPI, Request, HTTPException, File, UploadFile, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
import argparse
import os
import numpy as np
import cv2
import base64

from ultralytics import YOLO, checks
import ffmpeg

app = FastAPI(root_path=".")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://0.0.0.0:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('static/weight/best.pt')  # load a custom model

os.makedirs("static/predict/", exist_ok=True)

# Code copied from https://github.com/kkroening/ffmpeg-python/issues/246#issuecomment-520200981
def vidwrite(input, output, vcodec='libx264'):
    process = (
        ffmpeg
        .input(input)
        .output(output, vcodec=vcodec)
        .overwrite_output()
        .run()
    )
    process

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse('home.html', {"request": request})


@app.get("/about/")
def about_us(request: Request):
    '''
    Display about us page
    '''

    return templates.TemplateResponse('about.html', {"request": request})


@app.get("/video/")
def video(request: Request, link: str | None = None):
    '''
    Display about us page
    '''
    if link:
        get_model_result(model, link)

        filename = link.split('/')[-1].replace('?', '_').replace('=', '_')
        filename = filename + '.mp4'
        name = 'predict/' + filename
        return templates.TemplateResponse('video.html',
                                          {"request": request, 'filename': filename})

    return templates.TemplateResponse('video.html',
                                      {"request": request})


##############################################
# ------------POST Request Routes--------------
##############################################
@app.post("/")
async def detect_via_web_form(request: Request,
                              file: List[UploadFile] = File(...),
                              ):
    '''
    Requires an image file upload, model name (ex. yolov8n). Optional image size parameter (Default 640).
    Intended for human (non-api) users.
    Returns: HTML template render showing bbox data and base64 encoded image
    '''

    # create a copy that corrects for cv2.imdecode generating BGR images instead of RGB
    # using cvtColor instead of [...,::-1] to keep array contiguous in RAM
    # img_batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_batch]
    img_batch = [cv2.imdecode(np.fromstring(await file.read(), np.uint8), cv2.IMREAD_COLOR) for file in file]
    # img_batch = cv2.imdecode(np.fromstring(await file.read(), np.uint8), cv2.IMREAD_COLOR)

    results = model(img_batch, imgsz=640)

    box_values_list = results_values(results)

    return templates.TemplateResponse('show_results.html', {
        'request': request,
        'end': True,
        # unzipped in jinja2 template
        'box_values_list': box_values_list
    })


@app.post("/video/")
async def video(request: Request, file: UploadFile = File(...)):
    '''
    Requires an video file upload, max_size 2Mb
    '''

    content = await file.read()
    filename = file.filename

    if (len(content) > 10000000):
        return templates.TemplateResponse('video.html', {
            "request": request,
            "mensagem": "Apenas aquivos menores que 10MB."
        })

    dir_input = "static/input/"
    os.makedirs(dir_input, exist_ok=True)
    new_file = f"{dir_input + filename}"

    with open(new_file, "wb") as f:
        f.write(content)

    get_model_result(model, new_file)

    filename = f'{filename.split(".")[0]}.mp4'
    return templates.TemplateResponse('video.html',
                                      {"request": request,
                                       'filename': filename})


def get_model_result(model, path) -> None:
    for _ in model.predict(path, stream=True, save=True, name="", exist_ok=True, project=".", vid_stride=True):
        continue
    filename = filename.split('.')[0] + 'conv.mp4'
    vidwrite(filename, filename)


def results_values(results):
    r = []
    for result in results:
        img = result.plot()
        _, im_arr = cv2.imencode('.jpeg', img)
        im_b64 = base64.b64encode(im_arr.tobytes()).decode('utf-8')
        names = result.names
        boxes = result.boxes  # Boxes object for bbox outputs
        # masks = result.masks  # Masks object for segmentation masks outputs
        # probs = result.probs  # Class probabilities for classification outputs
        r.append({
            'im_b64': im_b64,
            'names': names[0],
            'boxes_conf': zip([x for x in boxes.xyxy.tolist()], [x for x in boxes.conf.tolist()]),
        })
    return r


async def iterable(file_output):  #
    with open(file_output, mode="rb") as f:  #
        yield f.read()


@app.get("/get_video/{video_path}")
def get_video(video_path: str):

    print(video_path)
    if video_path:
        filename = f'static/predict/{video_path}'

        file_stats = os.stat(filename)
        headers = {'Content-Disposition': f'attachment; filename="{video_path}.mp4"',
                   'accept-ranges': 'bytes',
                   'cache-control': 'no-cache',
                   'content-length': f'{file_stats.st_size}',
                   'content-range': f'bytes */{file_stats.st_size}'
                   }

        response = StreamingResponse(iterable(filename),
                                     status_code=206,
                                     headers=headers,
                                     media_type="video/mp4")
        return response
    else:
        return Response(status_code=404)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=8000)
    # parser.add_argument('--gpu', action='store_false', help="Choise GPU instance")
    opt = parser.parse_args()

    app_str = 'main:app'  # make the app string equal to whatever the name of this file is
    uvicorn.run(app_str, host=opt.host, port=opt.port, reload=True)

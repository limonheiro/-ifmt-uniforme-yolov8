from typing import List, Optional, Annotated, Generator, Any
from fastapi import FastAPI, Request, Form, File, UploadFile, Query
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

app = FastAPI(root_path="")
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

files = {
    item: os.path.join('static/infer/output/', item)
    for item in os.listdir('static/infer/output/')
}

model = YOLO('static/weight/best.pt')  # load a custom model


# Command line subprocess
# https://stackoverflow.com/a/29610897
def cmdline(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return str(process.communicate()[0])


color = (0, 200, 0)  # for bbox plotting


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
def about_us(request: Request, link: str | None = None):
    '''
    Display about us page
    '''
    if link:
        for i in model.predict(link, stream=True, save=True, name="", exist_ok=True, project="."):
            continue

        filename = link.split('/')[-1].replace('?', '_').replace('=', '_')
        filename = filename + '.mp4'
        name = 'predict/' + filename
        return templates.TemplateResponse('video.html',
                                          {"request": request, "link": link, 'filename': filename})

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
    format = ['asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', '.webm']

    if (len(content) > 2000000):
        return templates.TemplateResponse('video.html', {
            "request": request,
            "mensagem": "Apenas aquivos menores que 2MB."
        })

    return templates.TemplateResponse('video.html', {
        "request": request,
        "filename": str(filename_conv),
        "video": filename_conv
    })


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


@app.get("/get_video/{video_path}")
async def get_video(video_path: str):

    if video_path:
        def videotex(file_output: str) -> Generator[bytes, Any, None]:  #
            with open(file_output, mode="rb") as file_like:  #
                yield file_like.read()  #

        return FileResponse(f'predict/{video_path}', status_code=206, headers={
            'Content-Disposition': f'attachment;'}, filename="video_path",media_type="video/mp4",)
    else:
        return Response(status_code=404)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=8000)
    # parser.add_argument('--gpu', action='store_false', help="Choise GPU instance")
    opt = parser.parse_args()

    app_str = 'server:app'  # make the app string equal to whatever the name of this file is
    uvicorn.run(app_str, host=opt.host, port=opt.port, reload=True)

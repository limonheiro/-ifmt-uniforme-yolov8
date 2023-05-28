from typing import List, Any
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import uvicorn
import argparse
import numpy as np
import cv2

from ultralytics import YOLO
import ffmpeg
import os

from utils.result import results_values

from camera_multi import Camera

app = FastAPI(root_path="")
favicon_path = 'favicon.ico'
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

dir_predict = "static/predict/"

with open('static/weight/weight_name', 'r') as f:
    weight = f.readline()

model = YOLO(f'static/weight/{weight}')

##############################################
# -----------------Favicon--------------------
##############################################

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


############################################
# -----------------Webcam--------------------
#############################################

@app.get("/webcam/")
def webcam_livestream(request: Request):
    return templates.TemplateResponse("webcam.html", {
        'request': request,
        'webcam': True
    })


def gen(camera):
    """Video streaming generator function."""

    while True:
        try:
            frame = camera.get_frame()

            img_batch = cv2.imdecode(np.fromstring(frame, np.uint8), cv2.IMREAD_COLOR)
            img_batch_rgb = cv2.cvtColor(img_batch, cv2.COLOR_BGR2RGB)
            results = model(img_batch_rgb, imgsz=640)

            for r in results:
                img = r.plot()
                frame = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[1].tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n')

        except cv2.error as e:
            camera.video.release()
        except TypeError as e:
            camera.video.release()




@app.get('/video_feed', response_class=HTMLResponse)
async def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return StreamingResponse(gen(Camera()),
                             media_type='multipart/x-mixed-replace; boundary=frame')


##############################################
# ------------GET Request Routes--------------
##############################################


@app.get("/")
def home(request: Request):
    global model
    return templates.TemplateResponse('home.html', {"request": request})


@app.get("/about/")
def about_us(request: Request):
    '''
    Display about us page
    ok
    '''

    return templates.TemplateResponse('about.html', {"request": request})


@app.get("/video/")
async def video(request: Request, link: str | None = None):
    '''
    Display about us page
    '''

    if link:
        link = youtube_link(link)
        get_model_result(model, youtube_link(link))

        filename = link.split('/')[-1].replace('?', '_').replace('=', '_')

        out_file = f'{dir_predict}{filename}conv.mp4'
        filename = f'{dir_predict}{filename}.mp4'

        vidwrite(filename, out_file)

        delete_video_files([filename])

        return templates.TemplateResponse('video.html',
                                          {"request": request, 'filename': out_file.split('/')[-1],
                                           "video": True})

    return templates.TemplateResponse('video.html',
                                      {"request": request,
                                       "video": True})


@app.get("/get_video/{video_path}")
def get_video(video_path: str):
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
    img_batch = [cv2.imdecode(np.fromstring(await file.read(), np.uint8), cv2.IMREAD_COLOR) for file in file]
    results = model(img_batch, imgsz=640, save=True)

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

    if len(content) > 10000000:
        return templates.TemplateResponse('video.html', {
            "request": request,
            "mensagem": "Apenas aquivos menores que 10MB."
        })
    permit_video_format: list[str | Any] = ['asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv',
                                            'webm']

    if not (filename.split("/")[-1].split(".")[-1] in permit_video_format):
        return templates.TemplateResponse('video.html', {
            "request": request,
            "mensagem": f"formatos de video aceitos: {permit_video_format}"
        })

    dir_input = dir_predict
    os.makedirs(dir_input, exist_ok=True)
    new_file = f"{dir_input}{filename}"

    try:
        with open(new_file, "wb") as file:
            file.write(content)
    except Exception:
        return {"message": "There was an error uploading the file"}

    get_model_result(model, new_file)
    name = new_file.split(".")[0]

    out_file = f'{name}conv.mp4'
    filename = f'{name}.mp4'
    print(filename)
    vidwrite(filename, out_file)

    delete_video_files([filename, new_file])

    return templates.TemplateResponse('video.html',
                                      {"request": request,
                                       'filename': out_file.split('/')[-1],
                                       "video": True})


# @app.post("/webcam")
async def detect_via_webcam(file: UploadFile = File(...)):
    '''
    Requires an image file upload, model name (ex. yolov8n). Optional image size parameter (Default 640).
    Intended for human (non-api) users.
    Returns: HTML template render showing bbox data and base64 encoded image
    '''

    # create a copy that corrects for cv2.imdecode generating BGR images instead of RGB
    # using cvtColor instead of [...,::-1] to keep array contiguous in RAM
    file = await file.read()
    img_batch = cv2.imdecode(np.fromstring(file, np.uint8), cv2.IMREAD_COLOR)
    img_batch_rgb = cv2.cvtColor(img_batch, cv2.COLOR_BGR2RGB)
    results = model(img_batch_rgb, imgsz=640)

    box_values_list = results_values(results)

    return box_values_list[0]['im_b64']


def get_model_result(model, path) -> None:
    for _ in model.predict(path,
                           stream=True,
                           save=True,
                           name="predict",
                           exist_ok=True,
                           project="static/",
                           vid_stride=True,
                           line_width=1,
                           ):
        continue


async def iterable(file_output):  #
    with open(file_output, mode="rb") as f:  #
        yield f.read()


# Code copied from https://github.com/kkroening/ffmpeg-python/issues/246#issuecomment-520200981
def vidwrite(input: str, output: str, vcodec='libx264') -> None:
    process = (
        ffmpeg
        .input(input)
        .output(output, crf=25, vcodec=vcodec)
        .overwrite_output()
        .run()
    )
    process


def delete_video_files(file_list: list) -> None:
    for file in file_list:
        os.remove(file)
        if os.path.isfile(f'{file.split(".")[0]}.mp4'):
            os.remove(f'{file.split(".")[0]}.mp4')


def youtube_link(link: str) -> str:
    if "youtube" in link and not "watch?v=" in link:
        link = f'https://youtube.com/watch?v={link.split("/")[-1]}'
    elif "watch?v=" in link:
        link = f'https://youtube.com/watch?v={link.split("watch?v=")[-1]}'
    return link


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=8000)
    # parser.add_argument('--gpu', action='store_false', help="Choise GPU instance")
    parser.add_argument('--weight', default='bestn.pt')
    opt = parser.parse_args()

    with open('static/weight/weight_name', 'w') as f:
        f.write(opt.weight)

    app_str = 'main:app'  # make the app string equal to whatever the name of this file is
    uvicorn.run(app_str, host=opt.host, port=opt.port, reload=True)

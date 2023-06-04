import socket
from typing import List, Any
from fastapi import FastAPI, Request, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import uvicorn
import argparse
from pathlib import Path
import numpy as np
import cv2

from ultralytics import YOLO
import ffmpeg
import os
import websockets as webs

from utils.result import results_values

import glob

dir_predict = "static/predict/"

files = glob.glob(f'{dir_predict}*')
for f in files:
    os.remove(f)

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



with open('static/weight/weight_name', 'r') as f:
    weight = f.readline()

model = YOLO(f'static/weight/{weight}')

# https://github.com/ultralytics/ultralytics/blob/0d47d1139396496ab12ca004bead1021e0a4a6cf/ultralytics/yolo/utils/torch_utils.py#LL179C5-L179C99
yaml_file = getattr(model.model, 'yaml_file', '') or getattr(model.model, 'yaml', {}).get('yaml_file', '')
model_name = Path(yaml_file).stem.replace('yolo', 'YOLO') or model.info()
model_name = f'Model summary: {model_name[0]} layers, {model_name[1]} parameters, {model_name[2]} gradients' \
    if isinstance(model_name, tuple) else model_name


##############################################
# -----------------Favicon--------------------
##############################################

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


############################################
# -----------------Webcam--------------------
#############################################

@app.websocket("/ws")
async def get_stream(websocket: WebSocket):
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))  # depends on fourcc available camera
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 15)

    await websocket.accept()

    async def close_websocket():
        camera.release()
        await websocket.close(code=100)
        print("Client disconnected")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                close_websocket()
                break
            else:
                results = model(frame, imgsz=320)

                for r in results:
                    img = r.plot(line_width=1, font_size=1)
                    conf = r.boxes.conf.amin().item() if len(r.boxes.conf) else 0
                    # choice confidence above 50%
                    if conf > 0.5:
                        frame = cv2.imencode('.jpg', img)[1].tobytes()
                    else:
                        frame = cv2.imencode('.jpg', frame)[1].tobytes()

                await websocket.send_bytes(frame)

    except webs.exceptions.ConnectionClosedError:
        close_websocket()
    except Exception as e:
        print(e)
        close_websocket()


@app.get("/webcam/")
def webcam_livestream(request: Request):
    """
        Get Webcam server or stream video
    """
    return templates.TemplateResponse("webcam.html", {
        'request': request,
        'webcam': True,
        'model_name': model_name,
    })


##############################################
# ------------GET Request Routes--------------
##############################################


@app.get("/")
def home(request: Request):
    """
        Home Page
    """
    return templates.TemplateResponse('home.html', {"request": request, 'model_name': model_name})


@app.get("/about/")
def about_us(request: Request):
    """
        Display about us page
    """
    return templates.TemplateResponse('about.html', {"request": request, 'model_name': model_name})


@app.get("/video/")
async def video(request: Request, link: str | None = None):
    """
        Send a link, ex: YouTube, and return a predict video
    """

    if link:
        link = youtube_link(link)
        get_model_result(model, youtube_link(link))

        filename = link.split('/')[-1].replace('?', '_').replace('=', '_')

        out_file = f'{dir_predict}{filename}conv.mp4'
        filename = f'{dir_predict}{filename}.mp4'

        vidwrite(filename, out_file)

        delete_video_files([filename])

        print(out_file.split('/')[-1])

        return templates.TemplateResponse('video.html',
                                          {"request": request,
                                           'filename': out_file.split('/')[-1],
                                           "video": True,
                                           "model_name": model_name})

    return templates.TemplateResponse('video.html',
                                      {"request": request,
                                       "video": True,
                                       "model_name": model_name})


@app.get("/get_video/{video_path}")
def get_video(video_path: str):
    """
        Return convert video in predict directory
    """
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
    results = model(img_batch, imgsz=640)

    box_values_list = results_values(results)

    return templates.TemplateResponse("show_results.html", {
        "request": request,
        "end": True,
        # unzipped in jinja2 template
        "box_values_list": box_values_list,
        "model_name": model_name
    })


@app.post("/video/")
async def video(request: Request, file: UploadFile = File(...)):
    """
    Requires a video file upload, max_size 2Mb
    Return: video predict with bbox predict
    """

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
            "mensagem": f"formatos de video aceitos: {permit_video_format}",
            "model_name": model_name,
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
                                       "filename": out_file.split('/')[-1],
                                       "video": True,
                                       "model_name": model_name})


def get_model_result(model, path) -> None:
    """
        Return result by predict
    """
    project, name, _ = dir_predict.split('/')
    for _ in model.predict(path,
                           imgsz=128,
                           stream=True,
                           project=project,
                           name=name,
                           save=True,
                           exist_ok=True,
                           vid_stride=True,
                           line_width=1):
        continue


async def iterable(file_output):
    """
        return streaming video
    """
    with open(file_output, mode="rb") as f:  #
        yield f.read()


# Code copied from https://github.com/kkroening/ffmpeg-python/issues/246#issuecomment-520200981
def vidwrite(input: str, output: str, vcodec='libx264') -> None:
    """
        Convert video
    """
    process = (
        ffmpeg
        .input(input)
        .output(output, crf=25, vcodec=vcodec)
        .overwrite_output()
        .run()
    )
    process


def delete_video_files(file_list: list) -> None:
    """
        Delete upload user video
    """
    for file in file_list:
        os.remove(file)
        if os.path.isfile(f'{file.split(".")[0]}.mp4'):
            os.remove(f'{file.split(".")[0]}.mp4')


def youtube_link(link: str) -> str:
    """
        if URI input is invidiuos, or another front-ends YouTube, and Shorts
        Return: normal YouTube URI
    """
    if "youtube" in link and not "watch?v=" in link:
        link = f'https://youtube.com/watch?v={link.split("/")[-1]}'
    elif "watch?v=" in link:
        link = f'https://youtube.com/watch?v={link.split("watch?v=")[-1]}'
    return link


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--host', default='localhost', help="\b")
    parser.add_argument('--port', default=8000, help="\b")
    parser.add_argument('--weight',
                        default='v8n',
                        choices=['v8n', 'v8s'],
                        help="""
                        v8n=YOLOv8n,
                        v8s=YOLOv8s
                        """)

    opt = parser.parse_args()

    with open('static/weight/weight_name', 'w') as f:
        choices = {'v8n': 'bestn.pt',
                   'v8s': 'bests.pt'}
        f.write(choices[opt.weight])

    app_str = 'main:app'  # make the app string equal to whatever the name of this file is
    uvicorn.run(app_str,
                host=opt.host,
                port=int(opt.port),
                ws_ping_interval=1,
                ws_ping_timeout=1,
                timeout_keep_alive=1,
                workers=4)

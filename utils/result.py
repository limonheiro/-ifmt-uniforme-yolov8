import cv2
import base64


def results_values(results):
    r = []
    for result in results:
        img = result.plot(line_width=1, font_size=1)
        _, im_arr = cv2.imencode('.jpg', img)
        im_b64 = base64.b64encode(im_arr.tobytes()).decode('utf-8')
        names = result.names
        boxes = result.boxes  # Boxes object for bbox outputs
        # masks = result.masks  # Masks object for segmentation masks outputs
        # probs = result.probs  # Class probabilities for classification outputs
        r.append({
            'im_arr': im_arr,
            'im_b64': im_b64,
            'names': names[0],
            'boxes_conf': zip([x for x in boxes.xyxy.tolist()], [x for x in boxes.conf.tolist()]),
        })
    return r

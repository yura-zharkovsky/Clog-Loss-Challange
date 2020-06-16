import cv2 
import numpy as np
from IPython.display import HTML
from base64 import b64encode


def display_clip(path):  
    print(path)
    mp4 = open(path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    html = HTML("""
    <video controls>
          <source src="%s" type="video/mp4">
    </video>
    """ % data_url)
    return html


def read_clip(path):
    frames = []
    cap = cv2.VideoCapture(path)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame) 
        else:
            break
    cap.release()
    return np.stack(frames, axis=0)
import cv2 
import numpy as np
from IPython.display import HTML
from base64 import b64encode
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt

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


def read_clip(path, height=-1, width=-1):
    path = path.decode("utf-8") if type(path) == bytes else path
    frames = []
    cap = cv2.VideoCapture(path)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if width > 0 and height > 0:
                frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA) 
            frames.append(frame) 
        else:
            break
    cap.release()
    frames = np.stack(frames, axis=0)    
    return frames

def extract_mask(img, threshold=20):
    img = img-np.mean(img, axis=-1)[...,None]
    img = np.max(np.abs(img), axis=-1)
    img = (img > threshold)
    img = binary_fill_holes(img).astype(np.uint8)
    return img[...,None]


def extract_masks(imgs):
    masks = [extract_mask(img) for img in imgs]
    return np.stack(masks, axis=0)


def display_fused(frame, mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[...,None]
    mask = mask * 255
    img = np.concatenate([gray, gray, mask], axis=-1)
    plt.imshow(img)
    plt.show()
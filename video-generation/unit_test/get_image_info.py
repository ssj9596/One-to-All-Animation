import os
import sys
import numpy as np
import pandas as pd
import cv2 as cv
from pandarallel import pandarallel
import tarfile
from PIL import Image

pandarallel.initialize(nb_workers=50, progress_bar=True)
csv_file = sys.argv[1]


def read_info(path):
    tar_file = os.path.dirname(path)
    image_filename = os.path.basename(path)
    with tarfile.open(tar_file, 'r') as tar:
        with tar.extractfile(image_filename) as image_data:
            image = np.frombuffer(image_data.read(), np.uint8)
            img = cv.imdecode(image, cv.IMREAD_UNCHANGED).astype(np.uint8) #bgr
    height, width = img.shape[:2]
    aspect_ratio = height / width
    resolution = height * width
    num_frames = 1
    return num_frames, height, width, aspect_ratio, resolution


df = pd.read_csv(csv_file)
(
    df["num_frames"],
    df["height"],
    df["width"],
    df["aspect_ratio"],
    df["resolution"],
) = zip(*(df['path'].parallel_apply(read_info)))

import ipdb
ipdb.set_trace()

save_name = os.path.splitext(csv_file)[0] + '_info.csv'
df.to_csv(save_name, index=False)
import ipdb
ipdb.set_trace()

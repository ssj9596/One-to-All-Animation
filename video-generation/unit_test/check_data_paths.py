import os
import numpy as np
import pandas as pd
from pandarallel import pandarallel
import tarfile
from PIL import Image

def is_video_file(path):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.mpeg', '.mpg']
    ext = os.path.splitext(path)[1]
    return ext.lower() in video_extensions

def is_path_valid(path):
    if os.path.isfile(path):
        if is_video_file(path):
            pass
        else:
            try:
                image = Image.open(path).convert('RGB')
            except Exception as e:
                print(e, path)
                return False
    else:
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        if not dirname.endswith('.tar'):
            return False
        else:
            with tarfile.open(dirname) as tar:
                names = tar.getnames()
                if not basename in names:
                    return False
    return True

pandarallel.initialize(nb_workers=50, progress_bar=True)

df = pd.read_csv("train_config_video_0912/part_1.csv")
ret = df['path'].parallel_apply(
    is_path_valid
)
df_ = df[ret]
print('total:', len(df))
print('valid:', len(df_))
import ipdb
ipdb.set_trace()

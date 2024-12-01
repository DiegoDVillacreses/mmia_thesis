import os
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)

import shutil
import random
import numpy as np
import torch

def delete_directory_recursively(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        print(f"Directory {dir_path} deleted.")
    else:
        print(f"Directory {dir_path} not found.")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def cropp(l_norm, image):
    # Image dimensions
    W, H = image.size
    C_x, C_y, B_w, B_h = (l_norm[i] * W if i % 2 == 0 else l_norm[i] * H for i in range(4))
    T_x, T_y = C_x - (B_w / 2), C_y - (B_h / 2)
    bounding_box = (T_x, T_y, T_x + B_w, T_y + B_h)
    return image.crop(bounding_box)
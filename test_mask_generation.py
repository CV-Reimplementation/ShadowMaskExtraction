import os

import PIL.Image as Image
from skimage.filters import threshold_otsu
from skimage import filters
import numpy as np
from skimage.morphology import disk
from p_tqdm import p_map

def process(file):
    train_A_path = './dataset/RDD/train/input'
    train_C_path = './dataset/RDD/train/target'

    s_name = os.path.join(train_A_path, file)
    sf_name = os.path.join(train_C_path, file)

    s_img = Image.open(s_name).convert("L")
    s_img = np.array(s_img).astype(np.float32)
    sf_img = Image.open(sf_name).convert("L")
    sf_img = np.array(sf_img).astype(np.float32)
    diff = (np.asarray(sf_img, dtype='float32') - np.asarray(s_img, dtype='float32'))
    # diff[diff < 0] = 0
    L = threshold_otsu(diff)
    mask = np.float32(diff > L) * 255
    mask = filters.median(mask, disk(5))
    mask = Image.fromarray(mask).convert("L")
    # plt.imshow(mask)
    # plt.show()
    mask.save('./dataset/RDD/train/mask/' + file)


if __name__ == '__main__':
    train_A_path = './dataset/RDD/train/input'
    train_C_path = './dataset/RDD/train/target'

    os.makedirs('./dataset/RDD/train/mask/', exist_ok=True)

    root_path = os.listdir(train_A_path)

    p_map(process, root_path)
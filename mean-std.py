import os
import numpy as np
from PIL import Image

root = 'data/train_data/'

R, G, B = [], [], []

for i in os.listdir(root):
    img = np.array(Image.open(root + i).convert('RGB'))
    img = img / 255
    img_size = img.shape[0] * img.shape[1]
    R.append(img[:, :, 0].sum() / img_size)
    G.append(img[:, :, 1].sum() / img_size)
    B.append(img[:, :, 2].sum() / img_size)

R = np.array(R)
G = np.array(G)
B = np.array(B)

print(np.mean(R), np.mean(G), np.mean(B))
print(np.var(R), np.var(G), np.var(B))
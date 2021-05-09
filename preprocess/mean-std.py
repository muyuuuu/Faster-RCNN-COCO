import os
import numpy as np
import cv2
from tqdm import tqdm

root = '/home/liyanni/1307/ljw/train_data/'

R, G, B = [], [], []

for i in tqdm(os.listdir(root)):
    data = cv2.mean(cv2.imread(root + i))
    R.append(data[0] / 255)
    G.append(data[1] / 255)
    B.append(data[2] / 255)

R = np.array(R)
G = np.array(G)
B = np.array(B)

print(np.mean(R), np.mean(G), np.mean(B))
print(np.var(R), np.var(G), np.var(B))
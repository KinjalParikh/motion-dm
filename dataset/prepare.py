import os
import numpy as np
from tqdm.auto import tqdm

dir = "./HumanAct12"
dest = "./HumanAct12_unconstrained.npy"
metaf = "./meta.npy"
data = []

for file_name in tqdm(os.listdir(dir)):
    src = os.path.join(dir, file_name)
    mo = np.load(src)
    if mo.shape[0] >= 160:
        data.append(mo[[np.linspace(0, mo.shape[0]//2 - 1, 40, dtype=int)]])
        data.append(mo[[np.linspace(mo.shape[0] // 2, mo.shape[0] - 1, 40, dtype=int)]])
    elif mo.shape[0] >= 40:
        data.append(mo[[np.linspace(0, mo.shape[0]//2, 40, dtype=int)]])


data = np.asarray(data)
meta = []
for i in range(3):
    mean = data[:, :, i, :].mean()
    std = data[:, :, i, :].std()
    meta.append([mean, std])
    data[:, :, i, :] = (data[:, :, i, :] - mean)/std

with open(dest, 'wb') as f:
    np.save(f, np.asarray(data))
with open(metaf, 'wb') as f:
    np.save(f, np.asarray(meta))
print(np.load(dest).shape)
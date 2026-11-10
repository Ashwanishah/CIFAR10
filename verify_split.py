import numpy as np, torchvision
from sklearn.model_selection import train_test_split

SEED = 26007
ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
y = np.array(ds.targets)
idx = np.arange(len(y))
tr, va = train_test_split(idx, test_size=5000, stratify=y, random_state=SEED)

tr_saved = np.load("splits/train_idx.npy")
va_saved = np.load("splits/val_idx.npy")
print("train identical:", np.array_equal(tr, tr_saved))
print("val identical:  ", np.array_equal(va, va_saved))
print("val per class:  ", np.bincount(y[va]).tolist())

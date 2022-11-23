import librosa
import os
import numpy as np
import torch
import torch.utils.data as data

# for a conda problem: https://github.com/conda/conda/issues/7980
path = 'F:/pythonProject/'  # 'your path'

# number of classes: v1:30 v2:35
CLASSES = []
for _, dirs, _ in os.walk(os.path.join(path, 'train/')):
    # return of os.walk: root, dirs, files, train/test dataset have the same classes
    CLASSES = dirs
    break
# print(CLASSES)

class SolarDataset(data.Dataset):

    def __init__(self, mode='train', root=path):
        super(SolarDataset, self).__init__()
        self.root = os.path.join(root, mode+'/')
        # print(self.root)  # Data comes from the directory path\train or path\test, according to mode
        self.data = list()
        self.prep_dataset()

    # read in "data" the file list
    def prep_dataset(self):
        for root, dirs, files in os.walk(self.root):
            for file in files:
                f_path, cmd = os.path.join(root, file), root.split('/')[-1]
                # print(root, root.split('/'))
                self.data.append((f_path, cmd))  # cmd refers to the content text
        # print(len(self.data))

    # obtain labeled MFCC inputs
    # read-only while indexing in the list data
    def __getitem__(self, idx):
        f_path, cmd = self.data[idx]
        x = self.transform(f_path)
        y = CLASSES.index(cmd)
        return x, y

    # .len() function
    def __len__(self):
        return len(self.data)

    # MFCC, sample rate=16kHz, 64 filters
    # win_len = 25ms, overlapping = 10ms => win_length = 25/1000*16000=400
    # hop_length = 15/1000*16000=240
    def transform(self, path, sr=16000):
        sig, sr = librosa.load(path, sr)
        spec = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=64, n_fft=400, hop_length=240)
        x = np.array(spec, np.float32, copy=False)
        # print(x.shape)  # (64, xx)
        x = torch.from_numpy(x)  # obtained Tensor x is not resizable
        return x


def _collate_fn(batch):
    inputs = torch.tensor([s[0].detach().numpy() for s in batch])  # all x in this batch, totally 16
    # convert to longtensor
    # print(type(batch[0][0]), type(inputs[0]))
    targets = [s[1] for s in batch]  # all indexes of cmd appeared in this batch


    B = len(batch)  # batch size
    F, T = inputs[0].shape

    # padding each x to the same length of 128
    # max_len = 0
    # for input in inputs:
    #     max_len = max(max_len, len(input[0]))
    temp = torch.zeros(B, F, 128)  # max_len)  # F*128 block repeat B times
    # print(temp.shape)
    for x in range(B):
        l_padding = (128-inputs[x].size(1))//2
        temp[x, :, l_padding:l_padding + inputs[x].size(1)] = inputs[x]  # symmetrical padding

    # inputs = temp.unsqueeze(1)
    targets = torch.LongTensor(targets)
    return inputs, targets

import torch
import numpy as np
from torch.utils.data import Dataset



class Mydataset(Dataset):
    def __init__(self, data, label, multichannel = False):
        data_x = torch.from_numpy(data)
        if  multichannel:
            pass
        else:    
            data_x = data_x.unsqueeze(1)
        data_y = np.array([int(i) for i in label])
        data_y = torch.from_numpy(data_y)

        self.data_x = data_x.to(torch.float32)
        self.data_y = data_y.long()
        self.len = data_y.shape[0]
    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]
    def __len__(self):
        return self.len

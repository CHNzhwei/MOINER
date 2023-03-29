import sys
sys.path.append("F:\\GitHub\\bidd-molmap-master\\")
import  numpy as np
import  pandas as pd
from utils.get_Mydataset import Mydataset

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from copy import copy
from tqdm import *
import numpy as np


class FindImportance():
    def __init__(self, X, Y, disease = "ACC", device = "cuda", df_grid = None):
        self.X = X
        self.Y = Y
        self.disease = disease
        self.device = device
        self.df_grid = df_grid  

    def get_model(self):
        rnet = torchvision.models.resnet152(pretrained=False)
        rnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        rnet.fc = nn.Linear(rnet.fc.in_features,2)
        rnet.to(self.device)
        rnet.load_state_dict(torch.load('F:\\IDRB\\张维\\课题\\ProMap\\Pytorch\\checkpoint\\checkpoint-22-3-15-RPPA-%s-1k.pt'%self.disease)) #,map_location=torch.device('cpu'))
        return rnet

    def predict(self,data_loader):
        Y_pred = []
        Y_true = []
        model = self.get_model()
        for _, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            model.eval()
            output = model(data)
            _, pred =  torch.max(output,1)
            for i in pred:
                Y_pred.append(i)
            for i in target:
                Y_true.append(i)
        mse = mean_squared_error(Y_true, Y_pred)
        return mse

    def ForwardPropFeatureImp(self):

        df_grid = self.df_grid.sort_values(['y', 'x']).reset_index(drop=True)
        dataset = Mydataset(data = self.X, label = self.Y)
        data_loader = DataLoader(dataset, batch_size = 64, shuffle= None)
        mse = self.predict(data_loader)
        results = []
        for i in tqdm(range(len(df_grid)), ascii= True):
            ts = df_grid.iloc[i]
            y = ts.y
            x = ts.x
            X1 = copy(self.X)
            X1[:, y, x] = np.zeros(X1[:, y, x].shape)
            dataset = Mydataset(data = X1, label = self.Y)
            data_loader = DataLoader(dataset, batch_size = 64, shuffle= None)
            mse_mutaion = self.predict(data_loader)
            res = mse_mutaion - mse  # if res > 0, important, othervise, not important
            results.append(res)
        S = pd.Series(results, name = 'importance')
        df = df_grid.join(S)
        return df
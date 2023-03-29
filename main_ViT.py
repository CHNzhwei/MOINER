import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import xgboost as xgb

from vit_pytorch import ViT
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from utils_model.get_Mydataset import Mydataset
from utils_model.earlystop import EarlyStopping
from utils_model.model_metrics import metrics
import warnings
warnings.filterwarnings("ignore")


def Data_loader(data, label):
    dataset = Mydataset(data, label, multichannel=True)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return data_loader

    
def binary_train(model, train_loader, device, lr):
    optimizer = optim.Adagrad(model.parameters(), lr = lr)
    model.train()
    optimizer.zero_grad()
    train_loss = []
    pred_score = []
    pred_label = []
    real_label = []
    for batch_index, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.squeeze().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label)
        # loss = F.binary_cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        score = F.softmax(output)
        _, pred =  torch.max(score,1)
        for i in pred:
            pred_label.append(i.detach().cpu())
        for i in label:
            real_label.append(i.detach().cpu())
        for i in score:
            pred_score.append(i[1].detach().cpu())
    
    metric = metrics(real_label, pred_label)
    train_acc, train_auc, train_precision, train_recall, train_f1_score, train_mcc = metric.binary_metrics(pred_score)
    return np.average(train_loss), train_acc

def binary_test(model, test_loder, device, save):
    model.eval()
    test_loss = []
    pred_score = []
    pred_label = []
    real_label = []
    for batch_index, (data, label) in enumerate(test_loder):
        data, label = data.to(device), label.squeeze().to(device)
        output = model(data)
        loss = F.cross_entropy(output, label)
        test_loss.append(loss.item())
        score = F.softmax(output)
        _, pred =  torch.max(score,1)
        for i in pred:
            pred_label.append(i.detach().cpu())
        for i in label:
            real_label.append(i.detach().cpu())
        for i in score:
            pred_score.append(i[1].detach().cpu())

    metric = metrics(real_label, pred_label)
    test_acc, test_auc, test_precision, test_recall, test_f1_score, test_mcc = metric.binary_metrics(pred_score)
    if save:
        return real_label, pred_label
    else:
        return np.average(test_loss), test_acc

def multicalss_train(model, train_loader, device, lr):
    optimizer = optim.Adagrad(model.parameters(), lr = lr)
    model.train()
    optimizer.zero_grad()
    train_loss = []
    pred_label = []
    real_label = []
    for batch_index, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.squeeze().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        score = F.softmax(output)
        _, pred =  torch.max(score,1)
        for i in pred:
            pred_label.append(i.detach().cpu())
        for i in label:
            real_label.append(i.detach().cpu())
    
    metric = metrics(real_label, pred_label)
    train_acc, train_f1_weight, train_f1_macro, train_cm, train_report = metric.multi_metrics()
    return np.average(train_loss), train_acc

def multicalss_test(model, test_loder, device, save):
    model.eval()
    test_loss = []
    pred_label = []
    real_label = []
    for batch_index, (data, label) in enumerate(test_loder):
        data, label = data.to(device), label.squeeze().to(device)
        output = model(data)
        loss = F.cross_entropy(output, label)
        test_loss.append(loss.item())
        score = F.softmax(output)
        _, pred =  torch.max(score,1)
        for i in pred:
            pred_label.append(i.detach().cpu())
        for i in label:
            real_label.append(i.detach().cpu())

    metric = metrics(real_label, pred_label)
    test_acc, test_f1_weight, test_f1_macro, test_cm, test_report = metric.multi_metrics()
    if save:
        return real_label, pred_label
    else:
        return np.average(test_loss), test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_class', default=2, type=int,
                        help="Input your number of sample class")
    parser.add_argument('--patch', default=10, type=int,
                        help="Input the size of the patch into which the image is segmented")
    parser.add_argument('--ensembl', default=0, type=int,
                        help="Choosing whether to use ensemble learning will significantly increase the training time of the model, but it does not guarantee that the performance will improve. 0: OFF, 1: ON.”")
    parser.add_argument('--note', default="none", type=str,
                        help="Add notes to your task")
    args = parser.parse_args()

    dataset = args.note
    patch = args.patch # 3
    lr = 5e-5 # 5e-5
    depth = 8 # 5e-5
    heads = 16 # 5e-5
    mlp_dim = 2048
    random_state = 0

    try:
        path = os.getcwd()
        model_output = path + '/results_classification/%s_rand_%s_patch_%s_lr_%s_depth_%s_heads_%s_mlp_%s'%(dataset, random_state, patch, lr, depth, heads, mlp_dim)
        os.makedirs(model_output)
    except:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs   = 10000
    patience = 100

    fold = 5 
    data  = np.load("./results_map/5.IE-MOIF_Transformed_Data_0.npy")
    label = np.load("./results_preprocessing/5.Data_label.npy")
    if data.shape[2] == data.shape[3]:
        pass
    else:
        zero_padding = np.zeros((data.shape[0],data.shape[1],data.shape[2],data.shape[2]-data.shape[3]))
        data = np.concatenate((data, zero_padding), axis=3)
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    ACC       = []
    F1        = []
    MCC       = []
    F1_weighted = []
    F1_macro    = []

    if args.ensembl == 0:
        for fold, (idx_train, idx_test) in enumerate(skf.split(data, label)):
            model = ViT(
                        channels    = data.shape[1],
                        image_size  = data.shape[2],
                        patch_size  = patch,
                        num_classes = int(args.n_class),
                        dim   = 1024,
                        depth = depth,
                        heads = heads,
                        mlp_dim = mlp_dim,
                        dropout = 0.1,
                        emb_dropout = 0.1).to(device)

            train_loader = Data_loader(data[idx_train], label[idx_train])
            test_loader = Data_loader(data[idx_test], label[idx_test])

            early_stopping = EarlyStopping(patience=patience, verbose=True, delta = 0.0001, path='%s/IE-MOIF_fold_%s.model'%(model_output,fold+1))
            
            for epoch in range(epochs):
                if args.n_class == 2:
                    t_loss, t_acc = binary_train(model, train_loader, device, lr = lr)
                    v_loss, v_acc = binary_test(model, test_loader, device, 0)
                else:
                    t_loss, t_acc = multicalss_train(model, train_loader, device, lr = lr)
                    v_loss, v_acc = multicalss_test(model, test_loader, device, 0)
                print("Fold:{} \t Train Epoch:{} \t Train Loss: {:.4f} \t Train Accuracy: {:.4f} \t Valid Loss: {:.4f} \t Valid ACC: {:.4f}".format(fold+1, epoch, t_loss, t_acc, v_loss, v_acc))
                    
                early_stopping(1 - v_acc, model)
                if early_stopping.early_stop:
                    print("Early stopping!!!")
                    break
            
            model.load_state_dict(torch.load('%s/IE-MOIF_fold_%s.model'%(model_output,fold+1)))
            if args.n_class == 2:
                test_real_label, test_pred_label = binary_test(model, test_loader, device, 1)
                ACC.append(accuracy_score(test_real_label, test_pred_label))
                F1.append(f1_score(test_real_label, test_pred_label))
                MCC.append(matthews_corrcoef(test_real_label, test_pred_label))
            else:
                test_real_label, test_pred_label = multicalss_test(model, test_loader, device, 1)

                ACC.append(accuracy_score(test_real_label, test_pred_label))
                F1_weighted.append(f1_score(test_real_label, test_pred_label, average='weighted'))
                F1_macro.append(f1_score(test_real_label, test_pred_label, average='macro'))
    else:
        for fold, (idx_train, idx_test) in enumerate(skf.split(data, label)):
            fold = fold+1
            train_loader = Data_loader(data[idx_train], label[idx_train])
            test_loader = Data_loader(data[idx_test], label[idx_test])

            train_pred = []
            test_pred  = []

            for block in range(9,13):
                model = ViT(
                            channels    = data.shape[1],
                            image_size  = data.shape[2],
                            patch_size  = patch,
                            num_classes = int(args.n_class),
                            dim   = 1024,
                            depth = block,
                            heads = heads,
                            mlp_dim = mlp_dim,
                            dropout = 0.1,
                            emb_dropout = 0.1).to(device)
                early_stopping = EarlyStopping(patience=patience, verbose=True, delta = 0.0001, path='%s/IE-MOIF_fold_%s_model_%s.model'%(model_output,fold,block))
                for epoch in range(epochs):
                    if args.n_class == 2:
                        t_loss, t_acc = binary_train(model, train_loader, device, lr = lr)
                        v_loss, v_acc = binary_test(model, test_loader, device, 0)
                    else:
                        t_loss, t_acc = multicalss_train(model, train_loader, device, lr = lr)
                        v_loss, v_acc = multicalss_test(model, test_loader, device, 0)
                    print("Fold:{} \t Model: {} Train Epoch:{} \t Train Loss: {:.4f} \t Train Accuracy: {:.4f} \t Valid Loss: {:.4f} \t Valid ACC: {:.4f}".format(fold, block, epoch, t_loss, t_acc, v_loss, v_acc))

                    early_stopping(1 - v_acc, model)
                    if early_stopping.early_stop:
                        print("Early stopping!!!")
                        break

                model.load_state_dict(torch.load('%s/IE-MOIF_fold_%s_model_%s.model'%(model_output,fold,block)))
                if args.n_class == 2:
                    train_real_label, train_pred_label = binary_test(model, train_loader, device, 1)
                    test_real_label, test_pred_label  = binary_test(model, test_loader, device, 1)
                    train_pred.append(train_pred_label)
                    test_pred.append(test_pred_label)
                    print("train acc: %s\t test acc: %s"%(accuracy_score(train_real_label, train_pred_label), accuracy_score(test_real_label, test_pred_label)))
                else:
                    train_real_label, train_pred_label = multicalss_test(model, train_loader, device, 1)
                    test_real_label, test_pred_label  = multicalss_test(model, test_loader, device, 1)
                    train_pred.append(train_pred_label)
                    test_pred.append(test_pred_label)

            if args.n_class == 2:
                LR = xgb.XGBClassifier(
                                    learning_rate=0.1, 
                                    n_estimators=500, 
                                    objective='binary:logistic')
                LR.fit(np.array(train_pred).T, label[idx_train])
                pred_label = LR.predict(np.array(test_pred).T)
                ACC.append(accuracy_score(label[idx_test], pred_label))
                F1.append(f1_score(label[idx_test], pred_label))
                MCC.append(matthews_corrcoef(label[idx_test], pred_label))
            else:
                LR = xgb.XGBClassifier(
                                    learning_rate=0.1, 
                                    n_estimators=500, 
                                    objective = 'multi:softproba')
                LR.fit(np.array(train_pred).T, label[idx_train])
                pred_label = LR.predict(np.array(test_pred).T)
                ACC.append(accuracy_score(label[idx_test], pred_label))
                F1_weighted.append(f1_score(label[idx_test], pred_label, average='weighted'))
                F1_macro.append(f1_score(label[idx_test], pred_label, average='macro'))
    print(len(ACC), len(F1), len(MCC))
    try:
        model_report = pd.DataFrame(
                                    {
                                    "Fold": [i for i in range(1,6)] ,
                                    "ACC": ACC,
                                    "F1": F1,
                                    "MCC": MCC,
                                    }).to_csv("%s/IE-MOIF_report_%s.csv"%(model_output,dataset), index=False)
    except:
        model_report = pd.DataFrame(
                                    {
                                    "Fold": [i for i in range(1,6)] ,
                                    "ACC": ACC,
                                    "F1_weighted": F1_weighted,
                                    "F1_macro": F1_macro,
                                    }).to_csv("%s/IE-MOIF_report_%s.csv"%(model_output,dataset), index=False)

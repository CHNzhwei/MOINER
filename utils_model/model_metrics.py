from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

class metrics():
    def __init__(self, 
                 y_ture,
                 y_pred,
                ):
        super().__init__()
        self.y_ture = y_ture
        self.y_pred = y_pred

    def binary_metrics(self, y_score):   
        acc       = accuracy_score(self.y_ture, self.y_pred)
        precision = precision_score(self.y_ture, self.y_pred)
        recall    = recall_score(self.y_ture, self.y_pred)
        f1        = f1_score(self.y_ture, self.y_pred)
        mcc       = matthews_corrcoef(self.y_ture, self.y_pred)
        auc       = roc_auc_score(self.y_ture, y_score)
        return acc, auc, precision, recall, f1, mcc

    def multi_metrics(self):
        acc       = accuracy_score(self.y_ture, self.y_pred)
        f1_weight = f1_score(self.y_ture, self.y_pred, average='weighted')
        f1_macro  = f1_score(self.y_ture, self.y_pred, average='macro')
        cm  = confusion_matrix(self.y_ture, self.y_pred)
        report = classification_report(self.y_ture, self.y_pred, digits=4)
        return acc, f1_weight, f1_macro, cm, report

    def plot_roc_curve(self, y_score):
        auc = roc_auc_score(self.y_ture, y_score)
        # FPR, TPR, thresholds = roc_curve(self.y_ture, y_score)
        # plt.figure(dpi=800)
        # plt.plot(FPR, TPR, color='red',label='ROC curve (area = %0.4f)' % auc)
        # plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        # plt.xlim([-0.05, 1.05])
        # plt.ylim([-0.05, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.legend(loc="lower right")
        # plt.savefig("./output/img/AUC-ROC Curve.jpg")


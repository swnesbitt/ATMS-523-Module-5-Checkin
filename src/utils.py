import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_curve, auc, precision_recall_curve

def threshold_metrics(y_true, y_prob, thr: float = 0.5):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return {"confusion_matrix": cm, "accuracy": acc, "precision": prec, "recall": rec, "y_pred": y_pred}

def roc_pr_curves(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    return (fpr, tpr, roc_auc), (prec, rec, pr_auc)

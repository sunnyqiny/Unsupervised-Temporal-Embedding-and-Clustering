#!/usr/bin/env python
# coding: utf-8


import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.misc import comb
from sklearn import metrics

    
class time_series_ucr(Dataset):
    """synthetic time series dataset from section 5.1"""
    
    def __init__(self,filename1):
        """
        Args:
            t0: previous t0 data points to predict from
            N: number of data points
            transform: any transformations to be applied to time series
        """
        self.transform = None

        train_data, train_label = self.load_data(filename1)
        self.x = torch.cat(train_data.shape[0]*[torch.arange(0, train_data.shape[1]).type(torch.float).unsqueeze(0)])
        self.fx = train_data
        self.label = train_label

        self.t0 = train_data.shape
        self.masks = self._generate_square_subsequent_mask(train_data)
        
    def __len__(self):
        return len(self.fx)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        
        sample = (self.x[idx,:],
                  self.fx[idx,:],
                  self.masks)
        
        if self.transform:
            sample=self.transform(sample)
            
        return sample
    
    def _generate_square_subsequent_mask(self, dataset):
        t0 = np.floor(dataset.shape[1] *0.9)

        t0 = t0.astype(int)
        mask = torch.zeros(dataset.shape[1], dataset.shape[1])
        for i in range(0,t0):
            mask[i,t0:] = 1 
        for i in range(t0,dataset.shape[1]):
            mask[i,i+1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))#.masked_fill(mask == 1, float(0.0))
        return mask

    def load_data(self, filename):
        data_label = np.loadtxt(filename, delimiter=',')
        data = data_label[:, 1:]
        label = data_label[:, 0].astype(np.int32)
        return data, label

def evaluation(prediction, label):
    acc = cluster_acc(label, prediction)
    nmi = metrics.normalized_mutual_info_score(label, prediction)
    ari = metrics.adjusted_rand_score(label, prediction)
    ri = rand_index_score(label, prediction)
    anmi = metrics.adjusted_mutual_info_score(label, prediction)
    # print('ri, nmi, acc, ari: ', (ri, nmi, acc, ari))
    return ri, nmi, acc, ari, anmi

def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
                for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    # assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def transfer_labels(labels):
    indexes = np.unique(labels)
    num_classes = indexes.shape[0]
    num_samples = labels.shape[0]

    for i in range(num_samples):
        new_label = np.argwhere(labels[i] == indexes)[0][0]
        labels[i] = new_label
    return labels, num_classes

#!/usr/bin/python
from operator import add
import numpy as np
import pandas as pd
import xgboost as xgb
import csv


def extract_data(filename):
  """ Extract data from file """
  with open(filename, 'rb') as f:
    reader=f.readlines()
    train_data_label = [[int(x) for x in line.split() if x.isdigit()] for line in reader] 
  # sorted by label
  train_data_label = sorted(train_data_label, key=lambda x: x[-1])
  train_data_label = np.array(train_data_label) 
  return train_data_label


def inlabel_shuffle(data):
  """ Shuffle begin half data and end half data """
  num_zero_data = np.sum(data[:,-1]==0)
  label_zero_data = data[:num_zero_data,:]
  label_one_data = data[num_zero_data:,:]
  np.random.shuffle(label_zero_data)
  np.random.shuffle(label_one_data)
  return data

def shuffle(data):
  np.random.shuffle(data)
  return data


def sampling(data, n_samples):
  """ Shuffle and Sampling """
  data = inlabel_shuffle(data)
  data = np.row_stack((data[:n_samples/2,:], data[-n_samples/2:,:]))
  return data


def train_model(data):
#  print('train model with sample size ' + str(n_samples) + ' now...')
  xg_train = xgb.DMatrix( data[:,:-1], label=data[:,-1]) 
  param = {'eta':0.05, 'max_depth': 3, 'silent':1,
           'min_child_weight':0, 'subsample':0.8, 'lambda':0.8,
           'scale_pos_weight': 1, 'objective':'binary:logistic'}
  param['eval_metric'] = ['rmse', 'auc']
  watchlist = [ (xg_train,'train') ]
  num_round = 500
  model = xgb.train(param, xg_train, num_round, watchlist,  verbose_eval=100 )
  return model



from sklearn import metrics
def calc_auc(y, pred_y):
  """ Calculate AUC giving true label y and predict probability pred_y """
  fpr, tpr, thresholds = metrics.roc_curve(y, pred_y)
  return metrics.auc(fpr, tpr)
  

def do_test(raw_data, model):
  sum_dauc = 0.0
  for k in range(20):
      # type one
      data = raw_data.copy()
      inlabel_shuffle(data[:,0:7])
      inlabel_shuffle(data[:,7:12])
      pred = model.predict(xgb.DMatrix(data[:,:-1]))  
      auc1 = calc_auc(data[:,-1], pred)
      data = raw_data.copy()
      inlabel_shuffle(data[:,:-1])
      pred = model.predict(xgb.DMatrix(data[:,:-1]))
      auc2 = calc_auc(data[:,-1], pred)
      sum_dauc += auc2 - auc1
  return sum_dauc



class Pair:
  def __init__(self, idx, auc):
    self.idx = idx
    self.auc = auc
 

import os
   

def model(filename):
    train_data_label = extract_data(filename)
    n_samples=3000
    print ('solving ' + filename + ' sample size: ' + str(n_samples))
    rank_list = list()
    with open('rank/'+filename[filename.rfind('/')+1:filename.rfind('.')]+'-'+str(n_samples),'w') as f:
      for s in range(5):
        # shuffle and sampling
        data = sampling(train_data_label, n_samples)

        total_auc=[0]*10
        for k in range(10):
          inlabel_shuffle(data[:-1,:])
          for j in range(5):
            tmp1=data[n_samples*j/10:n_samples*(j+1)/10,:]
            tmp2=data[n_samples*(j+5)/10:n_samples*(j+6)/10,:]
            t1=data[:n_samples*j/10,:]
            t2=data[n_samples*(j+1)/10:n_samples*(j+5)/10,:]
            t3=data[n_samples*(j+6)/10:,:]
            data_train=np.concatenate((t1,t2,t3))
            data_test=np.concatenate((tmp1,tmp2))
            model = train_model(data_train)
            total_auc[k]+=do_test(data_test, model)

        total_auc_baseline=[0]*10
        for k in range(10):
          shuffle(data[:-1,:])
          for j in range(5):
            tmp1=data[n_samples*j/10:n_samples*(j+1)/10,:]
            tmp2=data[n_samples*(j+5)/10:n_samples*(j+6)/10,:]
            t1=data[:n_samples*j/10,:]
            t2=data[n_samples*(j+1)/10:n_samples*(j+5)/10,:]
            t3=data[n_samples*(j+6)/10:,:]
            data_train=np.concatenate((t1,t2,t3))
            data_test=np.concatenate((tmp1,tmp2))
            model = train_model(data_train)
            total_auc_baseline[k]+=do_test(data_test, model)
        u=0
        for i in range(10):
          for j in range(10):
            if total_auc[i]>total_auc_baseline[j]:
              u+=1
        z=(u-50)/(10.0*10.0*21.0/12.0)**0.5
        #f.write(str(total_auc)+'\n')
        f.write(str(z)+'\n')


import datetime, sys
if __name__ == '__main__':
  start_time = datetime.datetime.now()
  filename = sys.argv[1]
  model(filename)
  end_time = datetime.datetime.now()
  print ('Escape time: ' + str(end_time - start_time))

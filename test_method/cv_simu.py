#!/usr/bin/python
import numpy as np
import xgboost as xgb


def swap_column(data, col1, col2):
  """ Swap two columns of a matrix """
  tmp = data[:,col1].tolist()
  data[:,col1] = data[:,col2]
  data[:,col2] = tmp
  return data

def extract_data(filename):
  """ Extract data from file """
  with open(filename, 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    reader.next()  # skip title
    train_data_label = [[int(x) for x in line] for line in reader] 
  # sorted by label
  train_data_label = sorted(train_data_label, key=lambda x: x[-1])
  train_data_label = np.array(train_data_label) 
  # swap column 0 with 48, 10 with 49
  train_data_label = swap_column(train_data_label, 0, 48)
  train_data_label = swap_column(train_data_label, 10, 49)
#  print (train_data_label[:,-1])
  return train_data_label

import csv


def preproc(dtrain, dtest, param):
  label = dtrain.get_label() 
  ratio = float(np.sum(label == 0)) / np.sum(label==1)
  param['scale_pos_weight'] = ratio
  return (dtrain, dtest, param)


def train_model(data,filename):
  xg_train = xgb.DMatrix( data[:,:-1], label=data[:,-1]) 
  param = {'eta':0.05, 'max_depth': 3, 'silent':1, 'min_child_weight':0, 'subsample':0.8, 'lambda':0.8,  'scale_pos_weight': 1, 'objective':'binary:logistic'}
  param['eval_metric'] = ['rmse', 'auc']
  watchlist = [ (xg_train,'train') ]
  num_round = 500
  re = xgb.cv(param, xg_train, num_round, nfold=5, verbose_eval=10, seed = 0, fpreproc = preproc)


def model(filename):
  data = extract_data(filename)
  train_model(data, filename)       


import datetime, sys
if __name__ == '__main__':
  start_time = datetime.datetime.now()
  filename = sys.argv[1]
  model(filename)
  end_time = datetime.datetime.now()
  print ('Escape time: ' + str(end_time - start_time))
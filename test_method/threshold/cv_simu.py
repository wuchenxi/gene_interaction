#!/usr/bin/python
import numpy as np
import xgboost as xgb


def extract_data(filename):
  """ Extract data from file """
  with open(filename, 'rb') as f:
    reader=f.readlines()
    train_data_label = [[int(x) for x in line.split() if x.isdigit()] for line in reader] 
  # sorted by label
  train_data_label = sorted(train_data_label, key=lambda x: x[-1])
  train_data_label = np.array(train_data_label) 
  return train_data_label



def preproc(dtrain, dtest, param):
  label = dtrain.get_label() 
  ratio = float(np.sum(label == 0)) / np.sum(label==1)
  param['scale_pos_weight'] = ratio
  return (dtrain, dtest, param)


def train_model(data,filename):
  xg_train = xgb.DMatrix( data[:,:-1], label=data[:,-1]) 
  param = { 'silent':1,'sample_type':'weighted','max_depth':10,
            'min_child_weight':0,
            'booster':'gbtree','tree_method':'exact',
            'objective':'binary:logistic'}
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

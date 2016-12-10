#!/usr/bin/python
import numpy as np
import pandas as pd
import xgboost as xgb
import csv
from operator import add

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
    train_data_label = list()
    for line in reader:
      train_data_label.append([float(x) for x in line])
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

 

def train_model(data,filename):
  xg_train = xgb.DMatrix( data[:,:-1], label=data[:,-1])
  ratio = float(np.sum(data[:,-1] == 0)) / np.sum(data[:,-1]==1)
  param ={'eta':0.02, 'max_depth': 6, 'silent':1, 'min_child_weight':0,
           'subsample':0.8, 'lambda':0.8, 'scale_pos_weight':ratio,
           'objective':'binary:logistic'}
  param['eval_metric'] = ['rmse', 'auc']
  watchlist = [ (xg_train,'train') ]
  num_round = 500
  model = xgb.train(param, xg_train, num_round, watchlist,  verbose_eval=100)
  print ('write model to file now...')
  model.save_model('model/'+filename+'.model')
  #model.save_model(filename+'.model')
  return model


from sklearn import metrics
def calc_auc(y, pred_y):
  """ Calculate AUC giving true label y and predict probability pred_y """
  fpr, tpr, thresholds = metrics.roc_curve(y, pred_y)
  return metrics.auc(fpr, tpr)
  

def do_test(raw_data, model, starts, ends):
  n_genes = len(starts)
  delta_auc = list()
  for i in range(n_genes-1):
    for j in range(i+1, n_genes):
      sum_type_one = 0.0
      sum_type_two = 0.0  
      for k in range(100):
        # type one
        data = raw_data.copy()
        for l in range(n_genes): 
          inlabel_shuffle(data[:,starts[l]:ends[l]+1])
        pred = model.predict(xgb.DMatrix(data[:,:-1]))  
        auc = calc_auc(data[:,-1], pred)
##        print ('type one, k = ' + str(k) + ' auc: ' + str(auc))
        sum_type_one += auc 
        # type two
        data = raw_data.copy()
        ij = inlabel_shuffle(np.column_stack((data[:,starts[i]:ends[i]+1],data[:,starts[j]:ends[j]+1]))) 
        data[:,starts[i]:ends[i]+1] = ij[:,0:ends[i]+1-starts[i]]
        data[:,starts[j]:ends[j]+1] = ij[:,ends[i]+1-starts[i]:]
        for l in range(n_genes): # shuffle myself 
          if l != i and l != j:
            inlabel_shuffle(data[:,starts[l]:ends[l]+1])
        pred = model.predict(xgb.DMatrix(data[:,:-1]))
        auc = calc_auc(data[:,-1], pred)
##        print ('type two, k = ' + str(k) + ' auc: ' + str(auc))
        sum_type_two += auc 
      delta_auc.append((sum_type_two-sum_type_one)/100.0)
  return delta_auc


class Pair:
  def __init__(self, idx, auc):
    self.idx = idx
    self.auc = auc


def do_eval(delta_auc, n_genes, outfilename):
  """ Do evaluation """
  m = list()
  for i in range(0, n_genes-1):
    for j in range(i+1, n_genes):
      m.append("({0},{1})".format(i,j))
  rank = list()
  for i in range(len(delta_auc)):
    rank.append(Pair(i, delta_auc[i]))
  rank = sorted(rank, key=lambda x:x.auc, reverse=True)
  with open(outfilename,'w') as f:
    for i in range(len(rank)):
      f.write(m[rank[i].idx] + '\t' + str(rank[i].auc) + '\n') 
 

import os

def split_data(data):
  num_neg = np.sum(data[:,-1]==0)
  num_pos = np.sum(data[:,-1]==1)
  inlabel_shuffle(data)
  test_neg_start = num_neg/2
  test_neg_end = num_neg
  test_pos_start = num_neg+num_pos/2
  test_pos_end = num_neg+num_pos
  # splitting index 
  test_idx = range(test_neg_start, test_neg_end) + range(test_pos_start, test_pos_end)
  train_idx = range(0,test_neg_start)+range(test_neg_end,test_pos_start)
  return data[train_idx, :], data[test_idx, :]


def model(filename, starts, ends, outfilename):
  data = extract_data(filename)
  n_genes = len(starts)
  total_var = [0] * (n_genes * (n_genes - 1) /2)
  avg_diff = [0]*(n_genes * (n_genes - 1) /2)
  for i in range(0, 5):
    train_data, test_data = split_data(data)
    model = train_model(train_data, filename)       
    auc_1=do_test(test_data, model, starts, ends)
    model = train_model(test_data, filename)
    auc_2=do_test(train_data, model, starts, ends)
    if i==0:
      avg_diff=[(x+y)/2 for x, y in zip(auc_1, auc_2)]
    total_var=[x+0.1*(y-z)**2 for x, y, z in zip(total_var, auc_1, auc_2)]
  t_values=[x/(y**0.5) for x, y in zip(avg_diff, total_var)]
  do_eval(t_values, len(starts), outfilename)


def read_index(filename):
  with open(filename, 'rb') as f:
    r = csv.reader(f)
    line = r.next()
    starts = [int(x) for x in line]
    line = r.next()
    ends = [int(x) for x in line]
  return starts, ends


import datetime, sys
if __name__ == '__main__':
  start_time = datetime.datetime.now()
  filename = sys.argv[1]	# data file
  idx_filename = sys.argv[2]	# index file
  outfilename = sys.argv[3]	# output file
  starts, ends = read_index(idx_filename)
  model(filename, starts, ends, outfilename)
  end_time = datetime.datetime.now()
  print ('Escape time: ' + str(end_time - start_time))

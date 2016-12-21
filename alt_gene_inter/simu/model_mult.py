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



def inlabel_shuffle(data):
  """ Shuffle begin half data and end half data """
  num_zero_data = np.sum(data[:,-1]==0)
  label_zero_data = data[:num_zero_data,:]
  label_one_data = data[num_zero_data:,:]
  np.random.shuffle(label_zero_data)
  np.random.shuffle(label_one_data)
  return data


def extract_data(filename):
  """ Extract data from file """
  with open(filename, 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    train_data_label = [[float(x) for x in line] for line in reader]
  train_data_label = np.array(train_data_label)
  swap_column(train_data_label,10,48)
  swap_column(train_data_label,20,49)
  return train_data_label



def sampling(data, n_samples):
  """ Shuffle and Sampling """
  return data[np.random.randint(data.shape[0], size=n_samples), :]
   
 

def preproc(dtrain, dtest, param):
  label = dtrain.get_label() 
  ratio = float(np.sum(label == 0)) / np.sum(label==1)
  param['scale_pos_weight'] = ratio
  return (dtrain, dtest, param)


def train_model(data):
#  print('train model with sample size ' + str(n_samples) + ' now...')
  xg_train = xgb.DMatrix( data[:,:-1], label=data[:,-1])
  label = xg_train.get_label()
  ratio = float(np.sum(label==0)) / np.sum(label==1)	  

  param = {'eta':0.05, 'max_depth': 3, 'silent':1,
           'min_child_weight':0, 'subsample':0.8,
           'lambda':0.8, 'scale_pos_weight':ratio,
           'objective':'binary:logistic'}
  param['eval_metric'] = ['rmse','auc']
  watchlist = [ (xg_train,'train') ]
  num_round = 500
  model = xgb.train(param, xg_train, num_round, watchlist,
                    verbose_eval=100 )
  return model


from sklearn import metrics
def calc_auc(y, pred_y):
  """ Calculate AUC giving true label y and predict probability pred_y """
  fpr, tpr, thresholds = metrics.roc_curve(y, pred_y)
  return metrics.auc(fpr, tpr)
  

def do_test(raw_data, model, starts, ends):
  n_genes = len(starts)

  delta_auc = list()
  sample=raw_data[np.random.randint(raw_data.shape[0],size=2000), :]
  for i in range(n_genes-1):
    for j in range(i+1, n_genes):
        s=sample.copy()
        a=s[0:1000,:]
        b=s[1000:2000,:]
        c=a.copy()
        d=b.copy()
        e=a.copy()
        f=b.copy()
        g=a.copy()
        h=b.copy()
        c[:,starts[i]:ends[i]+1]=b[:,starts[i]:ends[i]+1]
        c[:,starts[j]:ends[j]+1]=b[:,starts[j]:ends[j]+1]
        d[:,starts[i]:ends[i]+1]=a[:,starts[i]:ends[i]+1]
        d[:,starts[j]:ends[j]+1]=a[:,starts[j]:ends[j]+1]
        e[:,starts[i]:ends[i]+1]=b[:,starts[i]:ends[i]+1]
        f[:,starts[i]:ends[i]+1]=a[:,starts[i]:ends[i]+1]
        g[:,starts[j]:ends[j]+1]=b[:,starts[j]:ends[j]+1]
        h[:,starts[j]:ends[j]+1]=a[:,starts[j]:ends[j]+1]
        sum1 = model.predict(xgb.DMatrix(a[:,:-1]))*model.predict(xgb.DMatrix(b[:,:-1]))*model.predict(xgb.DMatrix(c[:,:-1]))*model.predict(xgb.DMatrix(d[:,:-1])) 
        sum2 = model.predict(xgb.DMatrix(e[:,:-1]))*model.predict(xgb.DMatrix(f[:,:-1]))*model.predict(xgb.DMatrix(g[:,:-1]))*model.predict(xgb.DMatrix(h[:,:-1])) 
        #print i, j, np.sum(np.absolute(sum1-sum2))
        delta_auc.append(np.sum(np.absolute(np.log(sum1+0.0000001)-np.log(sum2+0.0000001))))
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
      print m[rank[i].idx], rank[i].auc
      f.write(m[rank[i].idx] + '\t' + str(rank[i].auc) + '\n') 


import os

def split_data(data, ratio):
  inlabel_shuffle(data)
  num_neg = np.sum(data[:,-1]==0)
  num_pos = np.sum(data[:,-1]==1)
  test_neg_start = 0
  test_neg_end = num_neg*ratio/10
  test_pos_start = num_neg
  test_pos_end = num_neg+num_pos*ratio/10
  # splitting index 
  test_idx = range(test_neg_start, test_neg_end) + range(test_pos_start, test_pos_end)
  train_idx = range(0,test_neg_start)+range(test_neg_end,test_pos_start) + range(test_pos_end, num_neg+num_pos)
  return data[train_idx, :], data[test_idx, :]

   

def model(filename, starts, ends, outfilename):
  data = extract_data(filename)
  n_genes = len(starts)
  total_auc = [0] * (n_genes * (n_genes - 1) /2)
  for i in range(10):
    train_data, test_data = split_data(data, 3) 
    model = train_model(train_data)       
    #print ("testing...")
    total_auc =map(add,total_auc,do_test(test_data, model, starts, ends))
  do_eval(total_auc, len(starts), outfilename)


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

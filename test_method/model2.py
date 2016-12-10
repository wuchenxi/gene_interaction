#!/usr/bin/python
from operator import add
import numpy as np
import pandas as pd
import xgboost as xgb
import csv

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
 

def preproc(dtrain, dtest, param):
  label = dtrain.get_label() 
  ratio = float(np.sum(label == 0)) / np.sum(label==1)
  param['scale_pos_weight'] = ratio
  return (dtrain, dtest, param)


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
#  print ('write model to file now...')
#  model.save_model('model/'+filename[filename.rfind('/')+1:filename.rfind('.')]+'-'+str(n_samples)+'.model')
  return model
  #re = xgb.cv(param, xg_train, num_round, nfold=5, verbose_eval=10, metrics={'rmse', 'auc'}, seed = 0, fpreproc = preproc)


from sklearn import metrics
def calc_auc(y, pred_y):
  """ Calculate AUC giving true label y and predict probability pred_y """
  fpr, tpr, thresholds = metrics.roc_curve(y, pred_y)
  return metrics.auc(fpr, tpr)
  

def do_test(raw_data, model):
  delta_auc = list()
  for i in range(4):
    for j in range(i+1, 5):
##      print ('solving i = {0}, j = {1}...'.format(i, j))
      sum_dauc = 0.0
      for k in range(20):
        # type one
        data = raw_data.copy()
        inlabel_shuffle(data[:,i*10:(i+1)*10])
        inlabel_shuffle(data[:,j*10:(j+1)*10])
        
        #uncomment the next 3 lines to shuffle the other 3 genes 
        for l in range(5): # shuffle myself 
          if l != i and l != j:
            inlabel_shuffle(data[:,l*10:(l+1)*10])
        
        pred = model.predict(xgb.DMatrix(data[:,:-1]))  
        auc1 = calc_auc(data[:,-1], pred)
##        print ('type one, k = ' + str(k) + ' auc: ' + str(auc)) 
        # type two
        data = raw_data.copy()
        ij = inlabel_shuffle(np.column_stack((data[:,i*10:(i+1)*10],data[:,j*10:(j+1)*10]))) 
        data[:,i*10:(i+1)*10] = ij[:,0:10]
        data[:,j*10:(j+1)*10] = ij[:,10:20]

        #uncomment the next 3 lines to shuffle the other 3 genes 
        for l in range(5): # shuffle myself 
          if l != i and l != j:
            inlabel_shuffle(data[:,l*10:(l+1)*10])
        
        pred = model.predict(xgb.DMatrix(data[:,:-1]))
        auc2 = calc_auc(data[:,-1], pred)
        sum_dauc += auc2 - auc1
      delta_auc.append(sum_dauc)
  return delta_auc



class Pair:
  def __init__(self, idx, auc):
    self.idx = idx
    self.auc = auc


def do_eval(delta_auc):
  """ Do evaluation """
##  print ('Do evaluation and write data...')
##  m = ['1 with 2', '1 with 3', '1 with 4', '1 with 5', '2 with 3', 
##       '2 with 4', '2 with 5', '3 with 4', '3 with 5', '4 with 5']
  rank = list()
  for i in range(10):
    rank.append(Pair(i, delta_auc[i]))
  rank = sorted(rank, key=lambda x:x.auc, reverse=True)
##  with open('rank/'+filename[filename.rfind('/')+1:filename.rfind('.')]+'-'+str(n_samples)+'.csv','w') as f:
##    for i in range(len(rank)):
##      f.write(m[rank[i].idx] + ',' + str(rank[i].auc) + '\n') 
  for i in range(len(rank)):
    if rank[i].idx == 0: return i
 

import os
   

def model(filename):
  train_data_label = extract_data(filename)
  for n_samples in range(1000, 2000, 1000):
    print ('solving ' + filename + ' sample size: ' + str(n_samples))
    rank_list = list()
    with open('rank/'+filename[filename.rfind('/')+1:filename.rfind('.')]+'-'+str(n_samples),'w') as f:
      for s in range(5):
        # shuffle and sampling
        data = sampling(train_data_label, n_samples)

        total_auc=[0]*10
        for j in range(5):
          tmp1=data[n_samples*j/10:n_samples*(j+1)/10,:]
          tmp2=data[n_samples*(j+5)/10:n_samples*(j+6)/10,:]
          t1=data[:n_samples*j/10,:]
          t2=data[n_samples*(j+1)/10:n_samples*(j+5)/10,:]
          t3=data[n_samples*(j+6)/10:,:]
          data_train=np.concatenate((t1,t2,t3))
          data_test=np.concatenate((tmp1,tmp2))
          model = train_model(data_train)
          total_auc=map(add,total_auc,do_test(data_test, model))

        total_auc_count=[0]*10
        for k in range(20):
          shuffle(data[:-1,:])
          total_auc_baseline=[0]*10
          for j in range(5):
            tmp1=data[n_samples*j/10:n_samples*(j+1)/10,:]
            tmp2=data[n_samples*(j+5)/10:n_samples*(j+6)/10,:]
            t1=data[:n_samples*j/10,:]
            t2=data[n_samples*(j+1)/10:n_samples*(j+5)/10,:]
            t3=data[n_samples*(j+6)/10:,:]
            data_train=np.concatenate((t1,t2,t3))
            data_test=np.concatenate((tmp1,tmp2))
            model = train_model(data_train)
            total_auc_baseline=map(add,total_auc_baseline,
                                   do_test(data_test, model))
          for i in range(10):
            if total_auc_baseline[i]>total_auc[i]:
              total_auc_count[i]+=1
            
        rank = do_eval(total_auc)
        f.write(str(rank)+'\n')
        f.write(str(total_auc_count)+'\n')


import datetime, sys
if __name__ == '__main__':
  start_time = datetime.datetime.now()
  filename = sys.argv[1]
  #model('dataset/0.025_0.3_0.4_0.4_0.15_data_EDM-10/0.025_0.3_0.4_0.4_0.15_data_EDM-10_1.txt') 
  model(filename)
  end_time = datetime.datetime.now()
  print ('Escape time: ' + str(end_time - start_time))

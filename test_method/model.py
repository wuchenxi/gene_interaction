#!/usr/bin/python
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
  train_data_label = np.asfarray(np.array(train_data_label)) 
  # swap column 0 with 48, 10 with 49
  train_data_label = swap_column(train_data_label, 0, 48)
  train_data_label = swap_column(train_data_label, 10, 49)
  #generate phenotype
  train_data_label[:,-1]=0.4*train_data_label[:,15]*train_data_label[:,34]+np.random.normal(size=train_data_label.shape[0])
  print "stdev=", np.var(0.4*train_data_label[:,15]*train_data_label[:,34])
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
  param = {'eta':0.05, 'max_depth': 3, 'silent':1, 'min_child_weight':0, 'subsample':0.8}
  param['eval_metric'] = ['rmse']
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
  sample=raw_data[np.random.randint(raw_data.shape[0], size=4000), :]
  for i in range(4):
    for j in range(i+1, 5):
        s=sample.copy()
        a=s[0:2000,:]
        b=s[2000:4000,:]
        c=a.copy()
        d=b.copy()
        e=a.copy()
        f=b.copy()
        g=a.copy()
        h=b.copy()
        c[:,i*10:(i+1)*10]=b[:,i*10:(i+1)*10]
        c[:,j*10:(j+1)*10]=b[:,j*10:(j+1)*10]
        d[:,i*10:(i+1)*10]=a[:,i*10:(i+1)*10]
        d[:,j*10:(j+1)*10]=a[:,j*10:(j+1)*10]
        e[:,i*10:(i+1)*10]=b[:,i*10:(i+1)*10]
        f[:,i*10:(i+1)*10]=a[:,i*10:(i+1)*10]
        g[:,j*10:(j+1)*10]=b[:,j*10:(j+1)*10]
        h[:,j*10:(j+1)*10]=a[:,j*10:(j+1)*10]
        sum1 = model.predict(xgb.DMatrix(a[:,:-1]))+model.predict(xgb.DMatrix(b[:,:-1]))+model.predict(xgb.DMatrix(c[:,:-1]))+model.predict(xgb.DMatrix(d[:,:-1])) 
        sum2 = model.predict(xgb.DMatrix(e[:,:-1]))+model.predict(xgb.DMatrix(f[:,:-1]))+model.predict(xgb.DMatrix(g[:,:-1]))+model.predict(xgb.DMatrix(h[:,:-1])) 
        delta_auc.append(np.sum(np.absolute(sum1-sum2)))
  print delta_auc

  return delta_auc
        
####      print ('solving i = {0}, j = {1}...'.format(i, j))
##      sum_type_one = 0.0
##      sum_type_two = 0.0  
##      for k in range(100):
##        # type one
##        data = raw_data.copy()
##        for l in range(5): # shuffle data: column [0,10),[10,20),[20,30),[30,40),[40,50)
##          inlabel_shuffle(data[:,l*10:(l+1)*10]) 
##        pred = model.predict(xgb.DMatrix(data[:,:-1]))  
##        auc = calc_auc(data[:,-1], pred)
####        print ('type one, k = ' + str(k) + ' auc: ' + str(auc))
##        sum_type_one += auc 
##        # type two
##        data = raw_data.copy()
##        ij = inlabel_shuffle(np.column_stack((data[:,i*10:(i+1)*10],data[:,j*10:(j+1)*10]))) 
##        data[:,i*10:(i+1)*10] = ij[:,0:10]
##        data[:,j*10:(j+1)*10] = ij[:,10:20]
##        for l in range(5): # shuffle myself 
##          if l != i and l != j:
##            inlabel_shuffle(data[:,l*10:(l+1)*10])
##        pred = model.predict(xgb.DMatrix(data[:,:-1]))
##        auc = calc_auc(data[:,-1], pred)
####        print ('type two, k = ' + str(k) + ' auc: ' + str(auc))
##        sum_type_two += auc 
##      delta_auc.append((sum_type_two-sum_type_one)/100.0)
##  print delta_auc
##  return delta_auc


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
def write_to_file(data, filename, n_samples, k):
  name = filename[filename.rfind('/')+1:filename.rfind('.')]
  folder_name = 'samples/' + name
  if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
  out_file = folder_name + '/' + name + '-' + str(n_samples) + '-' + ("%03d" % k) 
  with open(out_file, 'wb') as f:
    w = csv.writer(f, quoting=csv.QUOTE_NONE, delimiter='\t')
    for line in data:
      w.writerow(line)
   

def model(filename):
  train_data_label = extract_data(filename)
  for n_samples in range(5000, 6000, 1000):
    print ('solving ' + filename + ' sample size: ' + str(n_samples))
    rank_list = list()
    for k in range(1):
      # shuffle and sampling
      data = sampling(train_data_label, n_samples)
      write_to_file(data, filename, n_samples, k)
      model = train_model(data)       
      delta_auc = do_test(data, model)
      rank = do_eval(delta_auc)
      rank_list.append(rank)
    with open('rank/'+filename[filename.rfind('/')+1:filename.rfind('.')]+'-'+str(n_samples),'w') as f:
      for r in rank_list: f.write(str(r)+'\n') 


import datetime, sys
if __name__ == '__main__':
  start_time = datetime.datetime.now()
  filename = sys.argv[1]
  #model('dataset/0.025_0.3_0.4_0.4_0.15_data_EDM-10/0.025_0.3_0.4_0.4_0.15_data_EDM-10_1.txt') 
  model(filename)
  end_time = datetime.datetime.now()
  print ('Escape time: ' + str(end_time - start_time))

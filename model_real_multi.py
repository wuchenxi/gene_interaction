#!/usr/bin/python
import numpy as np
import xgboost as xgb
import csv
from operator import add
import datetime, sys
from multiprocessing import Process,Array,Lock


def swap_column(data, col1, col2):
  """ Swap two columns of a matrix """
  tmp = data[:,col1].tolist()
  data[:,col1] = data[:,col2]
  data[:,col2] = tmp
  return data

def extract_data(filename):
  """ Extract data from file """
  with open(filename, 'rb') as f:
    reader = csv.reader(f, delimiter=' ')
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


def train_model(data,filename,i):
  xg_train = xgb.DMatrix( data[:,:-1], label=data[:,-1]) 
  label = xg_train.get_label()
  ratio = float(np.sum(label==0)) / np.sum(label==1)	  
  param = {'eta':0.02, 'max_depth': 6, 'silent':1, 'min_child_weight':0, 'subsample':0.8, 'lambda':0.8, 'scale_pos_weight':ratio, 'objective':'binary:logistic'}
  param['eval_metric'] = ['rmse', 'auc']
  watchlist = [ (xg_train,'train') ]
  num_round = 500
  model = xgb.train(param, xg_train, num_round, watchlist,  verbose_eval=100)
  print ('write model to file now...')
  if not os.path.isdir('model'): os.mkdir('model')
  model.save_model('model/'+filename+str(i)+'.model')
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
   

def split_data(data, offset):
  num_neg = np.sum(data[:,-1]==0)
  num_pos = np.sum(data[:,-1]==1)
  test_neg_start = num_neg*offset/5
  test_neg_end = num_neg*(offset+1)/5
  test_pos_start = num_neg+num_pos*offset/5
  test_pos_end = num_neg+num_pos*(offset+1)/5
  # splitting index 
  test_idx = range(test_neg_start, test_neg_end) + range(test_pos_start, test_pos_end)
  train_idx = range(0,test_neg_start)+range(test_neg_end,test_pos_start) + range(test_pos_end, num_neg+num_pos)
  return data[train_idx, :], data[test_idx, :]

lock = Lock()

def g_add_to_total_auc(auc, total_auc):
  with lock:
    total_auc = map(add, total_auc, auc)


def test_process(data, filename, starts, ends, i,total_auc):
  start_time = datetime.datetime.now()
  train_data, test_data = split_data(data, i) 
  model = train_model(train_data, filename,i)       
  print ('testing in thread ', i)
  auc = do_test(test_data, model, starts, ends)
  g_add_to_total_auc(auc, total_auc)	# mutex
  end_time = datetime.datetime.now()
  print ('Escape time: ' + str(end_time - start_time) + ' of thread ' + str(i))


def model(filename, starts, ends, outfilename):
  data = extract_data(filename)
  n_genes = len(starts)
  total_auc = Array('d',[0] * (n_genes * (n_genes - 1) /2))
  process_list = []
  for i in range(0, 5): 
    t = Process(target=test_process,args=(data,filename,starts,ends,i,total_auc,))
    process_list.append(t)
  for t in process_list:
    t.start()
  for t in process_list: t.join()
  print ('Call do_eval now')
  do_eval(total_auc, len(starts), outfilename)


def read_index(filename):
  with open(filename, 'rb') as f:
    r = csv.reader(f)
    line = r.next()
    starts = [int(x) for x in line]
    line = r.next()
    ends = [int(x) for x in line]
  return starts, ends


if __name__ == '__main__':
  start_time = datetime.datetime.now()
  filename = sys.argv[1]	# data file
  idx_filename = sys.argv[2]	# index file
  outfilename = sys.argv[3]	# output file
  starts, ends = read_index(idx_filename)
  model(filename, starts, ends, outfilename)
  end_time = datetime.datetime.now()
  print ('Escape time: ' + str(end_time - start_time))

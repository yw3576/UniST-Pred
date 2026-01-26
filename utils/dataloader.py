import torch

import numpy as np
import os
from scipy.sparse import csr_matrix
import time
import yaml
import pickle
import random

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device

def simsfbay(data_dir, sequence_length, target_length):
    
    DEVICE = get_device()
    
    A_train = np.load(data_dir+'A_train.npy')
    
    inputs = np.load(data_dir+'train_t_simsf.npy')
    inputs = inputs.permute(0,2,1)
    targets = np.load(data_dir+'train_y_simsf.npy')
    targets = targets.permute(0,2,1)
    new_X = np.load(data_dir+'train_s_simsf.npy')
    
    X_pred = np.load(data_dir+'test_t_simsf.npy')
    X_pred = X_pred.permute(0,2,1)
    y_pred = np.load(data_dir+'test_y_simsf.npy')
    y_pred = y_pred.permute(0,2,1)
    new_X_pred = np.load(data_dir+'test_s_simsf.npy')
    
    ind_pred = round(len(inputs)*0.8)
    
    temp = list(zip(inputs[:ind_pred,:,:], targets[:ind_pred,:,:], new_X[:,ind_pred]))
    random.shuffle(temp)
    res1, res2, res3 = zip(*temp)
    res1, res2, res3 = list(res1), list(res2), list(res3)
    random.seed()
    
    inputs = []
    targets = []
    new_X = []

    del inputs
    del targets
    del new_X
    del temp

    sample_num = len(res1)
    ratio = 0.8
    ind = int(sample_num*ratio)

    X_train = res1[:ind]  # data shape: N, sequences, num of nodes
    X_train = np.array(X_train)
    X_test = res1[ind:]
    X_test = np.array(X_test)
    res1 = []

    new_X_train = np.array(res3[:ind])
    new_X_test = np.array(res3[ind:])
    res3 = []

    y_train = res2[:ind]
    y_train = np.array(y_train)
    y_test = res2[ind:]
    y_test = np.array(y_test)
    res2 = []
    
    return X_train, y_train, new_X_train, X_test, y_test, new_X_test, X_pred, y_pred, new_X_pred, A_train.to(DEVICE)

def pemsbay(data_dir, sequence_length, target_length):

    pems_lanes = np.load(data_dir+'pems_lanes.npy')

    with open(data_dir+'adj_mx_bay.pkl', 'rb') as f:
        adj_mx = pickle.load(f, encoding='latin1')

    adj_dense = torch.from_numpy(adj_mx[2]).float()

    # Alternatively, if you want a stacked 2D tensor of indices:
    indices = torch.nonzero(adj_dense).t() # .t() transposes to (2, num_non_zeros)
    values = torch.ones(indices.shape[1])
    for i in range(indices.shape[1]):
        values[i] = adj_dense[indices[0,i], indices[1,i]]

    adj_sparse_coo = torch.sparse_coo_tensor(indices, values)

    E_train = []
    E_train.append(adj_sparse_coo)
    #E_train = adj_sparse_coo
    num_nodes = 325

    DEVICE = get_device()
    #A_train = A_train.to(DEVICE)
    torch.cuda.empty_cache()

    A_train = []
    A_train_pred = []
    for i,edge in enumerate(E_train):
        #edge_index = np.array(edge._indices())
        edge_index = csr_matrix(edge.to_dense())
        edge_tmp = torch.from_numpy(np.vstack((edge_index.nonzero()[1], edge_index.nonzero()[0]))).type(torch.LongTensor)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.float)
        edge_tmp = edge_tmp.to(DEVICE)
        value_tmp = value_tmp.to(DEVICE)
        # normalize each adjacency matrix
        A_train.append((edge_tmp,value_tmp))
        A_train_pred.append((edge_tmp.to('cpu'), value_tmp.to('cpu')))
    edge_tmp = torch.stack((torch.arange(0,num_nodes),torch.arange(0,num_nodes))).type(torch.LongTensor)
    edge_tmp = edge_tmp.to(DEVICE)
    value_tmp = torch.ones(num_nodes).type(torch.float)
    value_tmp = value_tmp.to(DEVICE)
    A_train.append((edge_tmp.detach(),value_tmp.detach()))
    A_train_pred.append((edge_tmp.detach().to('cpu'),value_tmp.detach().to('cpu')))



    data = np.load(data_dir+'PEMS-bay.npy')

    shape = data.shape

    total_time = 52116
    X = []
    y = []
    for i in range(52116-sequence_length-target_length):
        data_e = data[i:i+sequence_length,:]
        target_e = data[i+sequence_length:i+target_length+sequence_length]
        X.append(data_e)
        y.append(target_e)

    inputs = np.array(X, dtype=np.uint8)
    targets = np.array(y, dtype=np.uint8)
    new_X= np.repeat(pems_lanes, inputs.shape[0], axis=0)
    
    ind_pred = round(len(inputs)*0.8)

    X_pred = inputs[ind_pred:, :, :]
    y_pred = targets[ind_pred:, :, :]
    new_X_pred = np.repeat(pems_lanes, X_pred.shape[0], axis=0)

    temp = list(zip(inputs[:ind_pred,:,:], targets[:ind_pred,:,:], new_X[:,ind_pred]))
    random.shuffle(temp)
    res1, res2, res3 = zip(*temp)
    res1, res2, res3 = list(res1), list(res2), list(res3)
    random.seed()
    
    inputs = []
    targets = []
    new_X = []

    del inputs
    del targets
    del new_X
    del temp

    sample_num = len(res1)
    ratio = 0.8
    ind = int(sample_num*ratio)

    X_train = res1[:ind]  # data shape: N, sequences, num of nodes
    X_train = np.array(X_train)
    X_test = res1[ind:]
    X_test = np.array(X_test)
    res1 = []

    new_X_train = np.array(res3[:ind])
    new_X_test = np.array(res3[ind:])
    res3 = []

    y_train = res2[:ind]
    y_train = np.array(y_train)
    y_test = res2[ind:]
    y_test = np.array(y_test)
    res2 = []
    
    return X_train, y_train, new_X_train, X_test, y_test, new_X_test, X_pred, y_pred, new_X_pred, A_train

def nyctaxi(data_dir, sequence_length = 35, target_length = 1):
    
    adj_mx = np.load(data_dir+'adj_mx.npz')['adj_mx']
    adj_dense = torch.from_numpy(adj_mx).float()


    # Alternatively, if you want a stacked 2D tensor of indices:
    indices = torch.nonzero(adj_dense).t() # .t() transposes to (2, num_non_zeros)
    values = torch.ones(indices.shape[1])

    adj_sparse_coo = torch.sparse_coo_tensor(indices, values)

    E_train = []
    E_train.append(adj_sparse_coo)
    num_nodes = 200


    DEVICE = get_device()

    A_train = []
    A_train_pred = []
    for i,edge in enumerate(E_train):
        #edge_index = np.array(edge._indices())
        edge_index = csr_matrix(edge.to_dense())
        edge_tmp = torch.from_numpy(np.vstack((edge_index.nonzero()[1], edge_index.nonzero()[0]))).type(torch.LongTensor)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.float)
        edge_tmp = edge_tmp.to(DEVICE)
        value_tmp = value_tmp.to(DEVICE)
        # normalize each adjacency matrix
        A_train.append((edge_tmp,value_tmp))
        A_train_pred.append((edge_tmp.to('cpu'), value_tmp.to('cpu')))
    edge_tmp = torch.stack((torch.arange(0,num_nodes),torch.arange(0,num_nodes))).type(torch.LongTensor)
    edge_tmp = edge_tmp.to(DEVICE)
    value_tmp = torch.ones(num_nodes).type(torch.float)
    value_tmp = value_tmp.to(DEVICE)
    A_train.append((edge_tmp,value_tmp))
    A_train_pred.append((edge_tmp.to('cpu'),value_tmp.to('cpu')))



    X_train = np.load(data_dir+'train.npz')['x'][:,:,:,0]
    y_train = np.load(data_dir+'train.npz')['y'][:,:,:,0]

    X_test = np.load(data_dir+'val.npz')['x'][:,:,:,0]
    y_test =np.load(data_dir+'val.npz')['y'][:,:,:,0]

    X_pred = np.load(data_dir+'test.npz')['x'][:,:,:,0]
    y_pred =np.load(data_dir+'test.npz')['y'][:,:,:,0]
    
    new_X_train = None
    new_X_test = None
    new_X_pred = None
    
    return X_train, y_train, new_X_train, X_test, y_test, new_X_test, X_pred, y_pred, new_X_pred, A_train

def DataLoader(data_name, data_dir, sequence_length, target_length):
    
    if data_name == 'SimSF-Bay':
        X_train, y_train, new_X_train, X_test, y_test, new_X_test, X_pred, y_pred, new_X_pred, A_train = simsfbay(data_dir, sequence_length, target_length)
    
    if data_name == 'pemsbay':
        X_train, y_train, new_X_train, X_test, y_test, new_X_test, X_pred, y_pred, new_X_pred, A_train = pemsbay(data_dir, sequence_length, target_length)
    
    if data_name == 'nyctaxi':
        X_train, y_train, new_X_train, X_test, y_test, new_X_test, X_pred, y_pred, new_X_pred, A_train = nyctaxi(data_dir, sequence_length, target_length)
    
    return X_train, y_train, new_X_train, X_test, y_test, new_X_test, X_pred, y_pred, new_X_pred, A_train

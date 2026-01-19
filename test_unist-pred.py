# -*- coding: utf-8 -*-

import torch

import numpy as np
import os
from scipy.sparse import csr_matrix

import argparse

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

torch.cuda.empty_cache()
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--remove_self_loops", action='store_true', default=True, help="remove_self_loops")
args = parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument('-ep', "--epochs", type=str, default='50') 
parser.add_argument('-batch', "--batch_size", type=str, default='64') 
parser.add_argument('-used', "--used_link", type=str, default='325')
parser.add_argument('-sequence', "--sequence_length", type=str, default='2016')
parser.add_argument("--remove_self_loops", action='store_true', default=True, help="remove_self_loops")

parser.add_argument('-channel', "--num_channels", type=str, default='4') 
parser.add_argument('-layer', "--num_layers", type=str, default='2') 
parser.add_argument('-w', "--w_out", type=str, default='100') 
parser.add_argument('-feat', "--feat_mixing_hidden_channels", type=str, default='100') 
parser.add_argument('-mixer', "--no_mixer_layers", type=str, default='4') 

args = parser.parse_args()


EPOCHS = int(args.epochs)
batch_size = int(args.batch_size)
num_link = int(args.used_link)
sequence_length = int(args.sequence_length)
path = args.data_path

num_channels = int(args.num_channels)
num_layers = int(args.num_layers)
w_out = int(args.w_out)
feat_mixing_hidden_channels = int(args.feat_mixing_hidden_channels)
no_mixer_layers = int(args.no_mixer_layers)

target_length = 12

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#from load_data_gnn_new_new_features import load_data_GNN
#features, new_features, edges, targets, new_link_01, scenarios_total = load_data_GNN(sequence_length=sequence_length, flow_path='flow_data_random')

pems_lanes = np.load('pems_lanes.npy')

import pickle

with open('adj_mx_bay.pkl', 'rb') as f:
    adj_mx = pickle.load(f, encoding='latin1')
    
adj_dense = torch.from_numpy(adj_mx[2]).float()
    
# 2. Identify non-zero elements and extract indices and values
#indices = torch.nonzero(adj_dense, as_tuple=True) # Returns (row_indices, col_indices)
#values = adj_dense[indices]

# Alternatively, if you want a stacked 2D tensor of indices:
indices = torch.nonzero(adj_dense).t() # .t() transposes to (2, num_non_zeros)
values = torch.ones(indices.shape[1])
for i in range(indices.shape[1]):
    values[i] = adj_dense[indices[0,i], indices[1,i]]

# 3. Construct the sparse COO tensor
# The size is inferred from the original dense tensor's shape
adj_sparse_coo = torch.sparse_coo_tensor(indices, values)

E_train = []
E_train.append(adj_sparse_coo)
#E_train = adj_sparse_coo
num_nodes = num_link


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device

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

import random

data = np.load('PEMS-bay.npy')

shape = data.shape
logger.info(f"batch size: {batch_size}")

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



ind_pred = round(len(inputs)*0.8)

X_pred = inputs[ind_pred:, :, :]
y_pred = targets[ind_pred:, :, :]
new_X_pred = np.repeat(pems_lanes, X_pred.shape[0], axis=0)

#random.seed(100000)
temp = list(zip(inputs[:ind_pred,:,:], targets[:ind_pred,:,:]))
random.shuffle(temp)
res1, res2 = zip(*temp)
res1, res2 = list(res1), list(res2)
random.seed()

inputs = []
targets = []

del inputs
del targets
del temp

sample_num = len(res1)
#print(sample_num)
ratio = 0.8
ind = int(sample_num*ratio)

X_train = res1[:ind]  # data shape: N, sequences, num of nodes
X_train = np.array(X_train)
new_X_train = np.repeat(pems_lanes, X_train.shape[0], axis=0)
X_test = res1[ind:]
X_test = np.array(X_test)
new_X_test = np.repeat(pems_lanes, X_test.shape[0], axis=0)
res1 = []


y_train = res2[:ind]
y_train = np.array(y_train)
y_test = res2[ind:]
y_test = np.array(y_test)
res3 = []

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train), torch.tensor(new_X_train))
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test), torch.tensor(new_X_test))
validation_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size
)

batch_size_01 = batch_size


pred_dataset = torch.utils.data.TensorDataset(torch.tensor(X_pred), torch.tensor(y_pred), torch.tensor(new_X_pred))
pred_loader = torch.utils.data.DataLoader(
    pred_dataset,
    batch_size=batch_size_01,
    shuffle=False
)

num_edge_type_train = len(A_train)
new_link_01 = np.array(range(num_link), dtype='int')
train_node = torch.from_numpy(new_link_01).type(torch.LongTensor)
train_node = train_node.to(DEVICE)

#num_classes = 7607
num_classes = num_link
is_ppi = False


final_f1, final_micro_f1 = [], []
tmp = None

w_in = 2
from utils.model_unistpred import GTN_TSmixer
model = GTN_TSmixer(num_edge=np.shape(edge_index)[1],
            num_channels=num_channels,
            w_in = w_in,
            w_out = w_out,
            #num_class=target_length,
            num_class=1,
            num_layers=num_layers,
            #num_nodes=7607,
            num_nodes=num_link,
            args = args,
            input_length=sequence_length,
            forecast_length=target_length,
            #forecast_length=1,
            no_feats=num_link,
            feat_mixing_hidden_channels=feat_mixing_hidden_channels,
            no_mixer_layers=no_mixer_layers,
            dropout=0.2
            )


pretrained = False
if pretrained:    
    PATH  = 'model_pemsbay_new_e_se_1d_residual_smoothl1.pt'

    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.to(DEVICE)
    #print(model)
else:
    model.to(DEVICE)

logger.info(f"Model: {model}")

def predict():
    #model = model.to(DEVICE)
    for i in range(1):
        
        model.eval()
        out_test_all = []
        y_test_all = []
        with torch.no_grad():
            for vbatch_idx, vdata in enumerate(pred_loader):
                #print('predict', vbatch_idx)
                #vfeatures = vdata.x
                #vedges = vdata.edge
                #vy = vdata.y
                vfeatures = vdata[0]
                vfeatures = vfeatures.type(torch.float32)
                vfeatures = vfeatures.to(DEVICE)
                
                vy = vdata[1]
                vy = vy.type(torch.float32)
                vy = vy.to(DEVICE)
                
                vfeatures_new = vdata[2]
                vfeatures_new = vfeatures_new.type(torch.float32)
                vfeatures_new = vfeatures_new.to(DEVICE)
                
                vdata = [] 
                
                vloss=0

                out_test = model(A_train, vfeatures, vfeatures_new, train_node.detach(), eval=True)

                out_test_all.append(out_test.to('cpu'))
                y_test_all.append(vy.to('cpu'))
                
                vfeatures = []
                vfeatures_new = []
                
            
        out_test_all = torch.cat(out_test_all)
        y_test_all = torch.cat(y_test_all)
        out_test_all = out_test_all.cpu()
        y_test_all = y_test_all.cpu()
        
    vloss = torch.sqrt(loss_fn(torch.squeeze(y_test_all[:,:,:]), torch.squeeze(out_test_all[:,:,:])))
    logger.info(f"prediction loss. loss: {vloss}")
    vloss = (loss_fn1(torch.squeeze(y_test_all[:,:,:]), torch.squeeze(out_test_all[:,:,:])))
    logger.info(f"prediction loss. loss: {vloss}")   

    vloss_total = torch.zeros((target_length, 1))
    for t in range(target_length):
        vloss_total[t,0] = torch.sqrt((loss_fn(torch.squeeze(y_test_all[:,t,:]), torch.squeeze(out_test_all[:,t,:]))))
                    
    logger.info(f"prediction loss. loss: {vloss_total}")

    vloss_total = torch.zeros((target_length, 1))
    for t in range(target_length):
        vloss_total[t,0] = ((loss_fn1(torch.squeeze(y_test_all[:,t,:]), torch.squeeze(out_test_all[:,t,:]))))
                    
    logger.info(f"prediction loss. loss: {vloss_total}")


lr = 0.0005
weight_decay = 0.1
betas = (0.9, 0.95)
epsilon = 1.0e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
logger.info(f"learning rate: {lr}")

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 50], gamma=0.1)

def adjust_learning_rate(optimizer, epoch):
	global lr
	#lr1 = lr
	"""Sets the learning rate to the initial LR decayed 10 times every 10 epochs"""
	if 1==0:
		lr1 = 0.001
	if epoch<=5:
		lr1 = lr * (0.1 ** (epoch//5))
	if epoch>5:
	    lr1 = lr * (0.1)
    #lr1 = lr * 0.1
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr1

Ws = []

loss_fn = torch.nn.MSELoss()
loss_fn1 = torch.nn.L1Loss()
loss_fn2 = torch.nn.HuberLoss(delta=2.0)
beta = 2.0
logger.info(f"beta: {beta}")
loss_fn3 = torch.nn.SmoothL1Loss(beta=beta)

flag = 2
logger.info(f"flag: {flag}")

for i in range(EPOCHS):
    
    #adjust_learning_rate(optimizer, i)
    for param_group in optimizer.param_groups:
        learning_rate = param_group['lr']
     
    logger.info(f"epoch id. epoch index: {i}, learning rate: {learning_rate}")   
    running_loss = 0
    running_loss1 = 0
    model.train()
    
    for batch_idx, data in enumerate(train_loader):

        features = data[0]
        features = features.type(torch.float32)
        
        y = data[1]
        y = y.type(torch.float32)
        
        features_new = data[2]
        features_new = features_new.type(torch.float32)
        
        
        features = features.to(DEVICE)
        y = y.to(DEVICE)
        features_new = features_new.to(DEVICE)
        
        data = []
        
        
        loss = 0
            
        out_train = model(A_train, features, features_new, train_node.detach())
        loss = loss_fn3(torch.squeeze(y[:,:,:]), torch.squeeze(out_train[:,:,:]))
        shape1 = out_train.shape
        shape2 = y.shape
        optimizer.zero_grad()
        loss.backward()
        
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
            
        logger.info(f"batch index: {batch_idx}, batch loss: {loss}")
        running_loss = running_loss + loss.detach().to('cpu')
        
        features = []
        y = []
        
    running_loss = running_loss/((batch_idx+1))
    logger.info(f"training loss. loss: {running_loss}")
    
    model.eval()
    running_vloss = 0
    running_vloss1 = 0
    v_idx = 0
    with torch.no_grad():
        for vbatch_idx, vdata in enumerate(validation_loader):
            vfeatures = vdata[0]
            vfeatures = vfeatures.type(torch.float32)
                       
            vy = vdata[1]
            vy = vy.type(torch.float32)
            
            vfeatures_new = vdata[2]
            vfeatures_new = vfeatures_new.type(torch.float32)
            
            vfeatures = vfeatures.to(DEVICE)
            vy = vy.to(DEVICE)
            vfeatures_new = vfeatures_new.to(DEVICE)
            
            vdata = []

            vloss=0
            out_test = model(A_train, vfeatures, vfeatures_new, train_node.detach(), eval=True)

            vloss = torch.sqrt(loss_fn(torch.squeeze(vy[:,:,:]), torch.squeeze(out_test[:,:,:])))
            vloss1 = loss_fn1(torch.squeeze(vy[:,:,:]), torch.squeeze(out_test[:,:,:]))
            

            logger.info(f"batch index: {vbatch_idx}, validation batch loss: {vloss}, rmse: {vloss1}")
            
            vfeatures = []
            vy = []
            
            running_vloss = running_vloss+vloss.detach().to('cpu')
            running_vloss1 = running_vloss1+vloss1.detach().to('cpu')
            v_idx = v_idx+1
        running_vloss = running_vloss/(vbatch_idx + 1)
        running_vloss1 = running_vloss1/(vbatch_idx + 1)
    #print('running_vloss', running_vloss)
    logger.info(f"validating loss. loss: {running_vloss1}, rmse: {running_vloss}")
    
    PATH  = 'model_pytorch_pemsbay_new_e_se_1d_residual.pt'

    model = model.to("cpu")
    torch.save(model.state_dict(), PATH)
    model = model.to(DEVICE)
    
    scheduler.step()
    
    predict()


del X_train
del X_test
del y_train
del y_test
del train_dataset
del train_loader
del validation_loader
del test_dataset
 
    
    
    
    


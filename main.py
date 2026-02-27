# -*- coding: utf-8 -*-

import torch

import numpy as np
import os
from scipy.sparse import csr_matrix
import time
import yaml
import pickle
import random

import argparse

from model.model_unistpred import UniST_Pred
from utils.dataloader import dataloader

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device

def model_supervisor(args):

    DEVICE = get_device()
    EPOCHS = int(args.epochs)
    batch_size = int(args.batch_size)
    num_link = int(args.used_link)
    sequence_length = int(args.sequence_length)
    target_length = int(args.target_length)
    #path = args.data_path
    num_edge = args.num_edge
    loss_ = args.loss_fn

    num_channels = int(args.num_channels)
    num_layers = int(args.num_layers)
    w_out = int(args.w_out)
    feat_mixing_hidden_channels = int(args.feat_mixing_hidden_channels)
    no_mixer_layers = int(args.no_mixer_layers)
    data_name = args.name

    X_train, y_train, new_X_train, X_test, y_test, new_X_test, X_pred, y_pred, new_X_pred, A_train = dataloader(data_name, sequence_length, target_length)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.uint8), torch.tensor(y_train, dtype=torch.uint8), torch.tensor(new_X_train, dtype=torch.uint8))
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

    num_classes = num_link

    w_in = 2

    model = UniST_Pred(num_edge=num_edge,
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
    else:
        model.to(DEVICE)

    logger.info(f"Model: {model}")

    def predict():
        for i in range(1):
        
            model.eval()
            out_test_all = []
            y_test_all = []
            with torch.no_grad():
                for vbatch_idx, vdata in enumerate(pred_loader):
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
        
        vloss = torch.sqrt(loss_fn0(torch.squeeze(y_test_all[:,:,:]), torch.squeeze(out_test_all[:,:,:])))
        logger.info(f"prediction RMSE: {vloss}")
        vloss = (loss_fn1(torch.squeeze(y_test_all[:,:,:]), torch.squeeze(out_test_all[:,:,:])))
        logger.info(f"prediction mae: {vloss}")   

        vloss_total = torch.zeros((target_length, 1))
        for t in range(target_length):
            vloss_total[t,0] = torch.sqrt((loss_fn0(torch.squeeze(y_test_all[:,t,:]), torch.squeeze(out_test_all[:,t,:]))))
                    
            logger.info(f"prediction RMSE: {vloss_total}")

        vloss_total = torch.zeros((target_length, 1))
        for t in range(target_length):
            vloss_total[t,0] = ((loss_fn1(torch.squeeze(y_test_all[:,t,:]), torch.squeeze(out_test_all[:,t,:]))))
                    
            logger.info(f"prediction mae: {vloss_total}")


    lr =float(args.learning_rate)
    weight_decay = 0.1
    betas = (0.9, 0.95)
    epsilon = 1.0e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    logger.info(f"learning rate: {lr}")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 50], gamma=0.1)

    Ws = []
    
    loss_fn0 = torch.nn.MSELoss()
    loss_fn1 = torch.nn.L1Loss()
    loss_fn2 = torch.nn.HuberLoss(delta=2.0)
    beta = 2.0
    logger.info(f"beta: {beta}")
    loss_fn3 = torch.nn.SmoothL1Loss(beta=beta)
    
    if loss_==0:
        loss_fn = loss_fn0
    if loss_==1:
        loss_fn = loss_fn1
    if loss_==2:
        loss_fn = loss_fn2
    if loss_==3:
        loss_fn = loss_fn3

    flag = 2
    logger.info(f"flag: {flag}")

    for i in range(EPOCHS):

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
            loss = loss_fn(torch.squeeze(y[:,:,:]), torch.squeeze(out_train[:,:,:]))
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

                vloss = torch.sqrt(loss_fn0(torch.squeeze(vy[:,:,:]), torch.squeeze(out_test[:,:,:])))
                vloss1 = loss_fn1(torch.squeeze(vy[:,:,:]), torch.squeeze(out_test[:,:,:]))
            

                logger.info(f"batch index: {vbatch_idx}, validation rmse: {vloss}, mae: {vloss1}")
            
                vfeatures = []
                vy = []
            
                running_vloss = running_vloss+vloss.detach().to('cpu')
                running_vloss1 = running_vloss1+vloss1.detach().to('cpu')
                v_idx = v_idx+1
            running_vloss = running_vloss/(vbatch_idx + 1)
            running_vloss1 = running_vloss1/(vbatch_idx + 1)
        logger.info(f"validating rmse: {running_vloss}, mae: {running_vloss}")
    
        PATH  = 'model_pytorch_pemsbay_new_e_se_1d_residual.pt'

        model = model.to("cpu")
        torch.save(model.state_dict(), PATH)
        model = model.to(DEVICE)
    
        scheduler.step()
    
        predict()

 
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='configs/configure_pems.yaml', 
                    type=str, help='the configuration to use')
    args = parser.parse_args()
    
    print(f'Starting experiment with configurations in {args.config_filename}...')
    time.sleep(3)
    configs = yaml.load(
        open(args.config_filename), 
        Loader=yaml.FullLoader
    )

    args = argparse.Namespace(**configs)
    model_supervisor(args)    
    
    






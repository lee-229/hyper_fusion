import os
import datetime
import torch
from model import Hyper_DSNet
from models.HyperPNN import HyperPNN
from models.MTNet import TransBlock
from models.FusionNet import FusionNet
from models.MDA_Net import MDANet
from models.my_model import my_model_5_25
from models.HSpecNet import HSpecNet
from models.HyperTransformer import HyperTransformer
from models.TFNet import TFNet
from models.mytransformer import my_model_3_31_2
dataset = "FR1"
device = torch.device("cuda:0" )
parallel = True
device_ids = [0]
if dataset=="WDC":
    data_in_channel=191
elif dataset=="Pavia":
    data_in_channel=102
elif dataset=="Botswana":
    data_in_channel=145
elif dataset=="FR1":
    data_in_channel=69
train_dataset_path = 'data/Train_'+dataset+'.h5'
valid_dataset_path = 'data/Valid_'+dataset+'.h5'
MODEL  = MDANet(data_in_channel)
MODEL_name = "MDANet"  
TIMESTAMP=datetime.datetime.now().strftime('%y-%m-%d-%H')
if MODEL_name=='Hyper_DSNet':
    loss_type='L1'
    optimizer=torch.optim.Adam
    lr = 0.0001
    lr_policy = "fixed"
    gamma = ""
    stepsize=""
    weight_decay = 1e-7
    epochs = 2000
    ckpt = 50  # 每隔多少部存model
    batch_size = 32
    version = "2" 
    start_epoch = 0
elif MODEL_name=='HyperPNN':
    loss_type='L1'
    optimizer=torch.optim.Adam
    lr = 0.0001
    lr_policy = "fixed"
    gamma = ""
    stepsize=""
    weight_decay = 1e-7
    epochs = 2000
    ckpt = 50  # 每隔多少部存model
    batch_size = 32
    version = "1" 
    start_epoch = 0 
elif MODEL_name=='TransBlock':
    loss_type='L1'
    lr_policy = "fixed"
    optimizer=torch.optim.Adam
    lr = 2e-4
    weight_decay = 0
    epochs = 2000
    ckpt = 50  # 每隔多少部存model
    batch_size = 16
    version = "1" 
    start_epoch = 0    
elif MODEL_name=='FusionNet':
    loss_type='L2'
    optimizer=torch.optim.Adam
    lr = 3e-4
    weight_decay = 0
    epochs = 2000
    ckpt = 50  # 每隔多少部存model
    batch_size = 32
    version = "1" 
    start_epoch = 0  
elif MODEL_name=='MDANet':
    loss_type='L1'
    optimizer=torch.optim.Adam
    lr = 2e-4
    lr_policy = "step"
    gamma = 0.9
    stepsize=1800
    weight_decay = 0
    epochs = 450
    ckpt = 50  # 每隔多少部存model
    batch_size = 16
    version = "1" 
    start_epoch = 0 
elif MODEL_name=='my_model_5_25':
    loss_type='L1'
    optimizer=torch.optim.Adam
    lr = 2e-4
    lr_policy = "step"
    gamma = 0.9
    stepsize=1800
    weight_decay = 0
    epochs = 400
    ckpt = 50  # 每隔多少部存model
    batch_size = 16
    version = "1" 
    start_epoch = 0  
elif MODEL_name=='HSpecNet':
    loss_type='L1'
    optimizer=torch.optim.Adam
    lr = 1e-4
    lr_policy = "fixed"
    gamma =""
    stepsize=""
    weight_decay = 0
    epochs = 400
    ckpt = 50  # 每隔多少部存model
    batch_size = 32
    version = "4" 
    start_epoch = 0  
elif MODEL_name=='HyperTransformer':
    loss_type='L1'
    optimizer=torch.optim.Adam
    lr = 2e-4
    lr_policy = "fixed"
    gamma =""
    stepsize=""
    weight_decay = 0
    epochs = 400
    ckpt = 50  # 每隔多少部存model
    batch_size = 16
    version = "1" 
    start_epoch = 0 
elif MODEL_name=='TFNet':
    loss_type='L1'
    optimizer=torch.optim.Adam
    lr = 1e-4
    lr_policy = "fixed"
    gamma =""
    stepsize=""
    weight_decay = 0
    epochs = 2000
    ckpt = 50  # 每隔多少部存model
    batch_size = 32
    version = "1" 
    start_epoch = 400
elif MODEL_name=='my_model_3_31_2':
    loss_type='L1'
    optimizer=torch.optim.Adam
    lr = 1e-4
    lr_policy = "fixed"
    gamma =""
    stepsize=""
    weight_decay = 0
    epochs = 2000
    ckpt = 50  # 每隔多少部存model
    batch_size = 32
    version = "1" 
    start_epoch = 0
model_path = "./Weights/"+ dataset +"/"+ MODEL_name +"/"+version +"/"+str(start_epoch) +".pth"  # 模型参数存放地址
log_path = "./Logs/"+dataset+"/"+MODEL_name +"/"+version+"/"
log_file = log_path+"train_log.txt"
tensorboard_path = './tensorboard/'+MODEL_name+'_'+dataset+'_'+TIMESTAMP
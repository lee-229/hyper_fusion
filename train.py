import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from loss import super_loss
from torch.utils.data import DataLoader
from data import Dataset_Pro
from model import Hyper_DSNet
from models.HyperPNN import HyperPNN
import numpy as np
import argparse
import shutil
from torch.utils.tensorboard import SummaryWriter
import datetime
from Q_sam_ergas import compute_index
from mmcv import Config#MMCv的核心组件 config类

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID "

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True
cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='pan-sharpening implementation')
    parser.add_argument('-c', '--config', required=True, help='config file path')
    return parser.parse_args()




def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}



# summaries(model, grad=True)  ## Summary the Network   训练时把整个网络的结构、参数量打印出来

# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)   # learning-rate update  学习率的更新
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.5)

# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-7)



def save_checkpoint(model, epoch,model_out_path):  # save model function  保存模型
    
    torch.save(model.state_dict(), model_out_path)


def ergas_loss_function(input, target, img_number):
    input = input.permute([0, 2, 3, 1])  # NHWC
    target = target.permute([0, 2, 3, 1])
    ergas_value = np.zeros(img_number)
    sam_value = np.zeros(img_number)
    # ERGAS = np.zeros(1) #biancheng ndarray not float
    # SAM = np.zeros(1)
    for i in range(img_number):  # i = 0123
        mynet = input[i, :, :, :]  # 128 128 191
        ref = target[i, :, :, :]  # 128 128 191
        sam, ergas = compute_index(ref, mynet, 4)
        ergas_value[i] = ergas.float()
        sam_value[i] = sam.float()

    ERGAS = torch.from_numpy(ergas_value).float()
    ERGAS = torch.mean(ERGAS).to(cfg.device)
    ERGAS.requires_grad_()

    # SAM[0] = np.mean(sam_value)
    # SAM = torch.from_numpy(SAM).float()
    return ERGAS


def train(cfg,training_data_loader, validate_data_loader, start_epoch=0):
    print('Start training...')
    
    MODEL  = cfg.MODEL
    MODEL_name = cfg.MODEL_name

    model = MODEL  # pannet从model里读出来
    criterion=super_loss(cfg.loss_type).to(cfg.device) 
    optimizer = cfg.optimizer(model.parameters(), lr=cfg.lr) 
    if cfg.parallel:
            model = nn.DataParallel(model,cfg.device_ids)
            #criterion = nn.DataParallel(criterion,cfg.device_ids)
    model.to(cfg.device)
    criterion.to(cfg.device)
    print('lr:{}'.format(cfg.lr))
    print(get_parameter_number(model))
    
    if not os.path.exists(cfg.log_path):
         os.makedirs(cfg.log_path)
    
    with open(cfg.log_file, mode='a') as filename:
        filename.write('\n')  # 换行
        filename.write('lr:{}  version:{}  start_epoch:{}  MSELoss  Model_2c  batch_size = {}  Adam_weight_decay=0  parameter_number:{}'.format(cfg.lr,cfg.version,cfg.start_epoch,cfg.batch_size,get_parameter_number(model)))
        filename.write('\n')  # 换行
    if not os.path.exists(cfg.tensorboard_path):
         os.makedirs(cfg.tensorboard_path)
    writer = SummaryWriter(cfg.tensorboard_path)  ## Tensorboard_show: case 2
    
    if os.path.isfile(cfg.model_path):  # 如果有一些预训练的model可以直接调用

        weight = torch.load(cfg.model_path,map_location=cfg.device) # load Weights!
        model.load_state_dict(weight)
        print('Network is Successfully Loaded from %s' % cfg.model_path)
    
    starttime = datetime.datetime.now()
    print(starttime)
    
    
    if cfg.lr_policy=="step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = cfg.stepsize, gamma = cfg.gamma )
    

    for epoch in range(cfg.start_epoch, cfg.epochs, 1):  # epochs决定每个样本迭代的次数
        print("the %d epoch's learning_rate='%f" % (epoch, optimizer.param_groups[0]['lr']))
        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):  # iteration就是分批数，循环完以后每批数据都迭代了一次

            gt, lms, ms, pan, edge_pan = batch[0].to(cfg.device), batch[1].to(cfg.device), batch[2].to(cfg.device), batch[3].to(cfg.device), batch[4].to(cfg.device)
            
            
            optimizer.zero_grad()
            if cfg.MODEL_name=="Hyper_DSNet":
                output1 = model(pan, edge_pan, lms, ms)
            elif cfg.MODEL_name=="HyperPNN":
                output1 = model(pan,lms, ms)
            elif cfg.MODEL_name=="HSpecNet" or cfg.MODEL_name=="TFNet" or cfg.MODEL_name == "my_model_3_31_2":
                    output1 = model( pan,lms)
            elif cfg.MODEL_name=="FusionNet" or cfg.MODEL_name=="MDANet" or cfg.MODEL_name=="my_model_5_25":
                output1 = model( pan,lms) # version 1
                #output1 = model( pan,lms,ms) # version 4
            elif cfg.MODEL_name=="HyperTransformer":
                output1 = model( ms,pan,lms)
            loss = criterion(output1, gt)  # compute loss
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

            optimizer.zero_grad()
            loss.backward(torch.ones(loss.shape).to("cuda:0"))
            # loss.sum().backward()  # fixed
            optimizer.step()  # fixed
            if cfg.lr_policy=="step":
                scheduler.step()
            #print('epoch:[{}/{}] batch:[{}/{}] G_loss:{:.5f} '.format(epoch, epochs, iteration, len(training_data_loader),loss))

            # for name, layer in model.named_parameters():
            #     writer.add_histogram('net/' + name + '_data_weight_decay', layer, epoch * iteration)

        # lr_scheduler.step()  # if update_lr, activate here!

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        writer.add_scalar('L1_loss/t_loss', t_loss, epoch)  # write to tensorboard to check

        if epoch % cfg.ckpt == 0:  # if each ckpt epochs, then start to save model
            model_out_path =  "Weights/"+ cfg.dataset +"/"+ MODEL_name +"/"+ cfg.version+"/"
            if not os.path.exists(model_out_path):
                os.makedirs(model_out_path)
             # 模型参数存放地址
            save_checkpoint(model, epoch,model_out_path+str(epoch) +".pth")

        # ============Epoch Validate=============== #  验证 测试valid
        model.eval()
        ergas_loss = []

        with torch.no_grad():  # fixed
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, ms, pan, edge_pan = batch[0].to(cfg.device), batch[1].to(cfg.device), batch[2].to(cfg.device), batch[3].to(cfg.device), batch[
                    4].to(cfg.device)

                if cfg.MODEL_name=="Hyper_DSNet":
                    output1 = model(pan, edge_pan, lms, ms)
                elif cfg.MODEL_name=="HyperPNN":
                    output1 = model(pan,lms, ms)
                elif cfg.MODEL_name=="FusionNet" or cfg.MODEL_name=="MDANet"or cfg.MODEL_name=="my_model_5_25":
                    output1 = model( pan,lms) # version 1
                    #output1 = model( pan,lms,ms) # version 4
                elif cfg.MODEL_name=="HSpecNet"or cfg.MODEL_name=="TFNet" or cfg.MODEL_name == "my_model_3_31_2":
                    output1 = model( pan,lms)
                elif cfg.MODEL_name=="HyperTransformer":
                    output1 = model( ms,pan,lms)

                loss = criterion(output1, gt)

                ergas = ergas_loss_function(output1, gt, cfg.batch_size)
                ergas_loss.append(ergas.item())

                epoch_val_loss.append(loss.item())

        if epoch % 1 == 0:
            v_loss = np.nanmean(np.array(epoch_val_loss))
            v_ergas = np.nanmean(np.array(ergas_loss))
            writer.add_scalar('val/v_loss', v_loss, epoch)
            writer.add_scalar('v_ergas', v_ergas, epoch)
            print('Epoch: {}/{} training loss:{}  validate loss:{}  ergas:{}'.format(cfg.epochs, epoch, t_loss, v_loss, v_ergas))  # print loss for each epoch
            with open(cfg.log_file, mode='a') as filename:
                endtime = datetime.datetime.now()
                filename.write('Epoch: {}/{} training loss:{}  validate loss:{}  ergas:{}  time:{}'.format(cfg.epochs, epoch, t_loss, v_loss, v_ergas, endtime - starttime))
                filename.write('\n')  # 换行
            endtime = datetime.datetime.now()
            # print(endtime)
            # print(endtime - starttime)

    endtime = datetime.datetime.now()
    print("time:{}".format(endtime - starttime))
    writer.close()  # close tensorboard

    with open(cfg.log_path, mode='a') as filename:
        filename.write('\n')  # 换行


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)  ##读取配置文件
    
    train_set = Dataset_Pro(cfg.train_dataset_path)  # creat data for training   数据处理的函数，在data里面
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=cfg.batch_size, shuffle=True,
                                      pin_memory=False,
                                      drop_last=True)  # put training data to DataLoader for batches  按batch_size个数据分批

    validate_set = Dataset_Pro(cfg.valid_dataset_path)  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=cfg.batch_size, shuffle=True,
                                      pin_memory=False, drop_last=True)  # put training d ata to DataLoader for batches

    train(cfg,training_data_loader, validate_data_loader, cfg.start_epoch)  # call train function (call: Line 66)   分批以后的数据train

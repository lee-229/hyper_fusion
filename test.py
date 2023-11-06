import torch.nn.modules as nn
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from data import Dataset_Pro,Dataset_Pro_FR
from model import Hyper_DSNet
from models.HyperPNN import HyperPNN
from models.MDA_Net import MDANet
from models.TFNet import TFNet
from models.mytransformer import my_model_3_31_2
import h5py
import scipy.io as sio
import os
import datetime
from EdgeDetection import Edge
from Q_sam_ergas import my_compute_sam_ergas

device = torch.device("cuda:0" )
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

MODEL = MDANet
dataset = "FR1"
MODEL_NAME = 'MDANet'
version = '1'
parallel = True
epoch_list = list(range(7, 8))
epoch_list = [50*x for x in epoch_list]
for epoch in epoch_list:

    load_weight = "Weights/"+ dataset +"/" + MODEL_NAME  +"/"+version+"/"+ "{}.pth".format(epoch)  # chose model
    test_file_path = "data/Test_"+dataset+".h5"

    
    
    if dataset=="WDC":
        channel=191
        size = 128
        num_testing = 4
    elif dataset=="Pavia":
        channel=102
        size = 400
        num_testing = 2
    elif dataset=="Botswana":
        channel=145
        size = 128
        num_testing = 4
    elif dataset=="FR1":
        channel=69
        size = 240
        num_testing = 2

    test_set = Dataset_Pro_FR(test_file_path)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=num_testing)

    model = MODEL(channel)
    model = model.to(device)  # fixed, important!

    if parallel == True:
        weight = torch.load(load_weight,map_location=torch.device('cpu')) # load Weights!
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k  in weight:
            v = weight[k]
            k = k.replace('module.', '')
            new_state_dict[k]=v

        model.load_state_dict(new_state_dict)
    else:
         weight = torch.load(load_weight,map_location=device) # load Weights!
         model.load_state_dict(weight)
        
        


    output1 = np.zeros([num_testing, size, size, channel])
    # output2 = np.zeros([num_testing, size, size, channel])

    starttime = datetime.datetime.now()

    for iteration, batch in enumerate(testing_data_loader, 1): 
        if dataset=="FR1":
            lms, ms, pan = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        else:
            gt, lms, ms, pan = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)

            edge_pan = torch.from_numpy(Edge(pan.cpu().numpy())).to(device).float()
        with torch.no_grad():
            if  MODEL_NAME=="Hyper_DSNet" :
                outputi1 = model(pan, edge_pan, lms, ms)
            elif MODEL_NAME=="HyperPNN":
                outputi1 = model(pan,lms, ms)
            elif MODEL_NAME=="MDANet":
                if  version=='1':
                    outputi1 = model( pan,lms)
                else :
                    outputi1 = model( pan,lms,ms)

            elif MODEL_NAME=="TFNet" or MODEL_NAME=="my_model_3_31_2":
                outputi1 = model( pan,lms)
                # outputi1 = model( pan,lms)
            output1[:, :, :, :] = outputi1.permute([0, 2, 3, 1]).cpu().detach().numpy()   #output:numpy n*h*w*c
            if dataset !="FR1":
                gt = gt.permute([0, 2, 3, 1]).cpu().detach().numpy() 
                print(gt.shape)
            lms = lms.permute([0, 2, 3, 1]).cpu().detach().numpy() 
            pan = pan.permute([0, 2, 3, 1]).cpu().detach().numpy() 
            
            # output2[:, :, :, :] = outputi2.permute([0, 2, 3, 1]).cpu().detach().numpy()   #output:numpy
    if dataset !="FR1":
        sam, ergas = my_compute_sam_ergas(torch.from_numpy(output1), test_file_path, num_testing) #torch
    # sam2, ergas2 = my_compute_sam_ergas(torch.from_numpy(output2), test_file_path, num_testing) #torch

    endtime = datetime.datetime.now()
    print("time:{}".format((endtime - starttime)))
    print("epoch:{}  sam:{}   ergas:{}".format(epoch,sam,ergas))

    save_path = "outputs/"+ dataset +"/" + MODEL_NAME  +"/"+version+"/"
    save_path_GT = "outputs/"+ dataset +"/"  +"GT"+"/"
    save_path_LMS = "outputs/"+ dataset +"/"  +"LMS"+"/"
    save_path_PAN = "outputs/"+ dataset +"/"  +"PAN"+"/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path_GT):
        os.makedirs(save_path_GT)
    if not os.path.exists(save_path_LMS):
        os.makedirs(save_path_LMS)
    if not os.path.exists(save_path_PAN):
        os.makedirs(save_path_PAN)        
    save_name = save_path+"{}.mat".format(epoch)  # chose model
    x = 1
    # save_name_GT = save_path_GT+"{}.mat".format(x)  # chose model
    # save_name_LMS = save_path_LMS+"{}.mat".format(x)  # chose model
    # save_name_PAN = save_path_PAN+"{}.mat".format(x)  # chose model
    sio.savemat(save_name, {'output': output1})
    # sio.savemat(save_path_GT, {'GT': gt})
    # sio.savemat(save_name_LMS, {'LMS': lms})
    # sio.savemat(save_name_PAN, {'PAN': pan})

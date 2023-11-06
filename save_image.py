import gdal
import os
import scipy.io as scio
import numpy as np
import cv2

def read_img2(path, name):
    img = scio.loadmat(path)[name]  # 读取matlab数据 生成字典格式 H*W*C
    return img

def tiff_save_img(img,img_path,dataset):
    if img.shape[2] == 1:
        img = img.squeeze()
        img = img * 255
        img = img.astype('uint8')  # 从float转换成整形
       # tifffile.imsave(img_path, img)
        write_img(img_path, img) 
    else:
        N,M,c = img.shape[0],img.shape[1],img.shape[2]
        NM = N*M
        img = img * 255
        img= img.astype(np.uint8)
        if dataset=="Botswana":
            IMG = img[:, :, [10,15,70]]
        else:
            IMG = img[:, :, [20,40,60]]
        
        for i in range(3):
            print(i)
            b = (IMG[:,:,i].reshape(NM,1))
            hb,levelb=np.histogram(b, bins=b.max()-b.min(), range=None)
            chb = np.cumsum(hb)
            t1 =levelb[np.where(chb > 0.1 * NM)[0][1]]
            t2_0 = np.where(chb <0.99 * NM)
            t2 = levelb[t2_0[0][np.size(t2_0[0]) - 1]]
            b[b < t1] = t1
            b[b > t2] = t2
            b = (b-t1)/(t2-t1)
            IMG[:,:,i] = b.reshape(N,M) * 255
        
        if dataset=="WDC":
            cv2.imwrite(img_path, IMG[:, :, [0,1,2]]) #读入顺序为BGR 与波段顺序一致
        if dataset=="Pavia":
            cv2.imwrite(img_path, IMG[:, :, [0,1,2]]) 
        if dataset=="Botswana":
            cv2.imwrite(img_path, IMG[:, :, [0,1,2]]) 
    
def write_img(filename,  im_data):

    #判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    #判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    #创建文件
    driver = gdal.GetDriverByName("GTiff") 
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    # dataset.SetGeoTransform(im_geotrans)       #写入仿射变换参数
    # dataset.SetProjection(im_proj)          #写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data) #写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset 
if __name__ == '__main__':
    dataset = "Botswana"
   
    epoch_list = list(range(7, 8))
    epoch_list = [50*x for x in epoch_list]

    # save GT   
    MODEL_NAME = "GT"
    filepath = "./outputs" + "/"+ dataset+ "/"+MODEL_NAME + "/"
    savepath = "./output_fig"+ "/"+ dataset+ "/" + MODEL_NAME + "/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    filename = filepath+"{}.mat".format(1) 
    output = read_img2(filename,'GT')
    print(output.shape)
    for i in range(output.shape[0]):
            img_name = str(i) + '.tif'
            tiff_save_img(output[i], os.path.join(savepath, img_name),dataset)  # 先转换成numpy 再保存RGB
    # #save model image
    # version = "1"
    # MODEL_NAME = "MDANet"
    # filepath = "./outputs" + "/"+ dataset+ "/"+MODEL_NAME + "/"+ version+"/"
    # savepath = "./output_fig"+ "/"+ dataset+ "/" + MODEL_NAME + "/"+ version
    # if not os.path.exists(savepath):
    #     os.makedirs(savepath)
    # for epoch in epoch_list:
    #     print(epoch)
    #     filename = filepath+"{}.mat".format(epoch) 
    #     print(filename)
    #     output = read_img2(filename,'output')
    #     for i in range(output.shape[0]):
    #         img_name = str(i) + '.tif'
    #         tiff_save_img(output[i], os.path.join(savepath, img_name),dataset)  # 先转换成numpy 再保存RGB


            
            
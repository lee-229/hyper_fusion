import numpy as np
import cv2
def guideFilter(I, g, winSize, eps):
    """I:导入的图像， g:引导图像"""
    mean_I = cv2.boxFilter(I, ddepth=-1, ksize=winSize, normalize=1)  # I的均值平滑
    mean_g = cv2.boxFilter(g, ddepth=-1, ksize=winSize, normalize=1)  # g的均值平滑

    mean_gg = cv2.boxFilter(g * g, ddepth=-1, ksize=winSize, normalize=1)  # I*I的均值平滑
    mean_Ig = cv2.boxFilter(I * g, ddepth=-1, ksize=winSize, normalize=1)  # I*g的均值平滑

    var_g = mean_gg - mean_g * mean_g  # 方差
    cov_Ig = mean_Ig - mean_I * mean_g  # 协方差

    a = cov_Ig / (var_g + eps)  # 相关因子a
    b = mean_I - a * mean_g  # 相关因子b

    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=winSize, normalize=1)  # 对a进行均值平滑
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=winSize, normalize=1)  # 对b进行均值平滑

    out = mean_a * g + mean_b
    return out


def wmap(img1, img2):
    # 权重映射
    data = (img1 == np.maximum(img1, img2)) * 1.0
    return data

def guider_fusion(img1, img2):
    """ 基于导向滤波的融合"""
    img1 = cv2.imread(img1, 1)/255.0
    img2 = cv2.imread(img2, 1)/255.0
 
    # 基层
    base1 = cv2.boxFilter(img1, -1, (31, 31), normalize=1)
    base2 = cv2.boxFilter(img2, -1, (31, 31), normalize=1)

    # 细节层
    detail1 = img1 - base1
    detail2 = img2 - base2

    # 拉普拉斯滤波
    h1 = abs(cv2.Laplacian(img1, -1))
    h2 = abs(cv2.Laplacian(img2, -1))

    # 高斯滤波
    s1 = cv2.GaussianBlur(h1, ksize=(11, 11), sigmaX=5, sigmaY=5)
    s2 = cv2.GaussianBlur(h2, ksize=(11, 11), sigmaX=5, sigmaY=5)

    # 获取权重矩阵
    p1 = wmap(s1, s2)
    p2 = wmap(s2, s1)

    # 导向滤波
    eps1 = 0.3**2
    eps2 = 0.03**2
    wb1 = guideFilter(p1, img1, (8, 8), eps1)
    wb2 = guideFilter(p2, img2, (8, 8), eps1)
    wd1 = guideFilter(p1, img1, (4, 4), eps2)
    wd2 = guideFilter(p2, img2, (4, 4), eps2)

    # 权重归一化
    wbmax = wb1 + wb2
    wdmax = wd1 + wd2
    wb1 = wb1 / wbmax
    wb2 = wb2 / wbmax
    wd1 = wd1 / wdmax
    wd2 = wd2 / wdmax

    # 融合
    B = base1 * wb1 + base2 * wb2
    D = detail1 * wd1 + detail2 * wd2
    im = B + D

    return im
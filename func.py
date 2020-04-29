import numpy as np
import math
import matplotlib.pyplot as plt


def Convolve(N, F):
    """
    卷积函数 边缘采用对称补充元素的方式
    :param N: 是原图像的一个通道
    :param F: 卷积核
    :param dtype: 默认类型
    :return: 卷积后矩阵
    """
    ih, iw = N.shape
    fh, fw = F.shape
    O = np.zeros_like(N, 'float')
    N = np.pad(N, ((fh // 2, fh // 2), (fw // 2, fw // 2)), 'symmetric')
    U, L = [fh // 2, fw // 2]
    D, R = [fh - U, fw - L]
    for i in range(U, U + ih):
        for j in range(L, L + iw):
            O[i - U, j - L] = (N[i - U:i + D, j - L:j + R] * F).sum()
    return O


def TwoD_Convolve(I, F):
    """
    用两次一维卷积代替二维卷积
    :param I: 待卷积图像的一个通道
    :param F: 一维卷积核
    :return: 卷积后图像
    """
    N = I.copy()
    fw = F.shape[1]
    N = Convolve(N, F)
    N = Convolve(N, F.T)
    return N


def convolve(filter,mat,padding,strides):

    result = None
    filter_size = filter.shape
    mat_size = mat.shape
    if len(filter_size) == 2:
        if len(mat_size) == 3:
            channel = []
            for i in range(mat_size[-1]):
                pad_mat = np.pad(mat[:,:,i], ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
                temp = []
                for j in range(0,mat_size[0],strides[1]):
                    temp.append([])
                    for k in range(0,mat_size[1],strides[0]):
                        val = (filter*pad_mat[j*strides[1]:j*strides[1]+filter_size[0],
                                      k*strides[0]:k*strides[0]+filter_size[1]]).sum()
                        temp[-1].append(val)
                channel.append(np.array(temp))

            channel = tuple(channel)
            result = np.dstack(channel)
        elif len(mat_size) == 2:
            channel = []
            pad_mat = np.pad(mat, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
            for j in range(0, mat_size[0], strides[1]):
                channel.append([])
                for k in range(0, mat_size[1], strides[0]):
                    val = (filter * pad_mat[j * strides[1]:j * strides[1] + filter_size[0],
                                    k * strides[0]:k * strides[0] + filter_size[1]]).sum()
                    channel[-1].append(val)


            result = np.array(channel)


    return result



def downsample(img,step = 2):
    return img[::step,::step]

def GuassianKernel(sigma , dim):
    '''
    :param sigma: Standard deviation
    :param dim: dimension(must be positive and also an odd number)
    :return: return the required Gaussian kernel.
    '''
    temp = [t - (dim//2) for t in range(dim)]
    assistant = []
    for i in range(dim):
        assistant.append(temp)
    assistant = np.array(assistant)
    temp = 2*sigma*sigma
    result = (1.0/(temp*np.pi))*np.exp(-(assistant**2+(assistant.T)**2)/temp)
    return result

def getDoG(img,n,sigma0,S = None,O = None):
    '''
    :param img: the original img.
    :param sigma0: sigma of the first stack of the first octave. default 1.52 for complicate reasons.
    :param n: how many stacks of feature that you wanna extract.
    :param S: how many stacks does every octave have. S must bigger than 3.
    :param k: the ratio of two adjacent stacks' scale.
    :param O: how many octaves do we have.
    :return: the DoG Pyramid
    '''
    if S == None:
        S = n + 3
    if O == None:
        O = int(np.log2(min(img.shape[0], img.shape[1]))) - 3

    k = 2 ** (1.0 / n)
    sigma = [[(k**s)*sigma0*(1<<o) for s in range(S)] for o in range(O)]
    samplePyramid = [downsample(img, 1 << o) for o in range(O)]

    GuassianPyramid = []
    for i in range(O):
        GuassianPyramid.append([])
        for j in range(S):
            dim = int(6*sigma[i][j] + 1)
            if dim % 2 == 0:
                dim += 1
            GuassianPyramid[-1].append(convolve(GuassianKernel(sigma[i][j], dim),samplePyramid[i],[dim//2,dim//2,dim//2,dim//2],[1,1]))
    DoG = [[GuassianPyramid[o][s+1] - GuassianPyramid[o][s] for s in range(S - 1)] for o in range(O)]


    return DoG,GuassianPyramid

def OneD_Gaussian(sig):
    """
    获得一维线性高斯核
    :param sig: 高斯核参数
    :return:
    """
    dim = np.int(6*sig+1)
    if dim % 2 == 0:
        dim += 1
    linear_Gaussian_filter = [np.abs(t - (dim // 2)) for t in range(dim)]
    linear_Gaussian_filter = np.array(
        [[1 / (math.sqrt(2 * math.pi) * sig) * math.exp(t * t * -0.5 / (sig * sig)) for t in linear_Gaussian_filter]])
    linear_Gaussian_filter = linear_Gaussian_filter / linear_Gaussian_filter.sum()
    return linear_Gaussian_filter


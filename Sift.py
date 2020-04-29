import func  # func是我自己编写的函数库
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import cv2
from PIL import Image


class KEYPOINTS:
    def __init__(self):
        self.x, self.y, self.layer, self.sig, self.val = 0, 0, 0, 0, 0
        self.sigo = 0  #
        self.r = 0
        self.dir = 0
        self.o = 0


class SIFT:
    """
    Sift类
    输入图像和初始化参数
    可以从中得到DOG keypoints 特征描述符
    """

    def __init__(self, img, n=3, sigma=1.52, BinNum=36):
        """
        初始化函数
        :param img: 待处理图像
        :param n: 每组内可以计算关键点的层数
        :param sigma: 高斯核初始大小
        """
        self.ori = img
        self.Img = img
        if len(img.shape) >= 3:
            self.Img = img.mean(axis=-1)
        self.n, self.sigma = n, sigma
        self.BinNum = BinNum
        self.T = 0.04
        self.gam = 10
        self.the = 0.5 * self.T / (self.n *255)
        self.s = n + 3  # 每组层数
        self.o = int(math.log2(min(img.shape[0], img.shape[1]))) - 3  # 组数
        self.Gaussian_Filter = self.Get_All_Gaussian()  # 每层的高斯核
        self.G, self.DoG = self.Get_GDoG()  # GDOG所有图像
        self.KeyPoints = np.array(self.Cal_KeyPoints())  # 算关键点位置和方向
        self.descriptors = self.Cal_Descriptors() # 算关键点描述子

    def Get_All_Gaussian(self):
        """
        计算n个一维高斯核
        :return: 一个列表（因为长度不一致） 每一行是一个高斯核
        """
        k = np.math.pow(2, 1 / self.n)
        Gaussian_F = [func.OneD_Gaussian(self.sigma * k ** i) for i in range(self.s)]
        return Gaussian_F

    def Get_GDoG(self):
        """
        计算DOG图像
        :return: DOG
        """
        img0 = self.Img
        Gau, DoG = [], []
        for i in range(self.o):  # 一共有o组
            G = [func.TwoD_Convolve(img0, self.Gaussian_Filter[j]) for j in range(self.s)]
            Gau.append(G)
            img0 = G[-3][1::2, 1::2]  # 倒数第三幅为下一组的第一幅的下采样
            DoG.append([G[j + 1] - G[j] for j in range(self.s - 1)])
        return Gau, DoG

    def Is_Local_Extrema(self, onow, snow, x, y, max_step=5, border=5):
        """
        调整极值点位置
        :param onow: 当前组
        :param snow: 当前层
        :param x,y: 位置
        :param max_step: 最大迭代次数
        :param border: 边界处理
        :return:
        """
        key_p = KEYPOINTS()

        h = 1.0
        d1, d22, d3 = 0.5 * h, h, 0.25 * h
        img, i = self.DoG[onow][snow], 0
        while i < max_step:
            # 如果解超过边界 则返回空值
            if snow < 1 or snow > self.n or y < border or y >= img.shape[1] - border or \
                    x < border or x >= img.shape[0] - border:
                return None, None, None

            # 三个区域
            img, pre, nex = self.DoG[onow][snow], self.DoG[onow][snow - 1], self.DoG[onow][snow + 1]

            # 一阶导数 梯度
            dg = np.array([(img[x][y + 1] - img[x][y - 1]) * d1, (img[x + 1][y] - img[x - 1][y]) * d1,
                           (nex[x][y] - pre[x][y]) * d1])

            # 海森矩阵
            d2 = img[x][y] * 2
            dxx = (img[x][y + 1] + img[x][y - 1] - d2) * d22
            dyy = (img[x + 1][y] + img[x - 1][y] - d2) * d22
            dss = (nex[x][y] + pre[x][y] - d2) * d22
            dxy = (img[x + 1][y + 1] - img[x + 1][y - 1] - img[x - 1][y + 1] + img[x - 1][y - 1]) * d3
            dxs = (nex[x][y + 1] - nex[x][y - 1] - pre[x][y + 1] + pre[x][y - 1]) * 0.25 * d3
            dys = (nex[x + 1][y] - nex[x - 1][y] - pre[x + 1][y] + pre[x - 1][y]) * 0.25 * d3
            H = np.array([[dxx, dxy, dxs],
                          [dxy, dyy, dys],
                          [dxs, dys, dss]])

            X = - np.linalg.pinv(H) @ dg
            # 如果都小于0.5 说明解稳定了 退出循环
            if np.abs(X).max() < 0.5:
                break

            # 更新解 进入下一次迭代
            y, x, snow = int(np.round(y + X[0])), int(np.round(x + X[1])), int(np.round(snow + X[2]))

            i += 1

        if i >= max_step:
            return None, x, y
        if snow < 1 or snow > self.n or y < border or y >= img.shape[1] - border or \
                x < max_step or x >= img.shape[0] - border:
            return None, None, None

        # 响应过小就滤去
        ans = img[x][y] * h + 0.5 * (dg @ X)
        if np.abs(ans) * self.n < 0.04:
            return None, x, y
        key_p.val = np.abs(ans) * self.n

        # 利用Hessian矩阵的迹和行列式计算主曲率的比值
        tr, det = dxx + dyy, dxx * dyy - dxy * dxy
        if det <= 0 or tr * tr >= 12.1 * det:
            return None, x, y

        # 获得一个关键点
        key_p.x, key_p.y = (x + X[1]) * (1 << onow), (y + X[0]) * (1 << onow)
        key_p.o = onow
        key_p.layer = snow
        key_p.sigo = self.sigma * np.power(2.0, (snow + X[2]) / self.n)
        key_p.r = int(np.round(3 * 1.52 * key_p.sigo))  # 半径

        return key_p, x, y

    def Get_MainDir(self, kp, x, y):
        '''
        获得主方向
        :param kp: 关键点信息
        :param x: x坐标（调整后的）
        :param y: y坐标
        :return:
        '''
        signow = 1.52 * kp.sigo
        exp_scale = -1.0 / (2.0 * signow * signow)
        img = self.G[kp.o][kp.layer]

        # 投票点的 DXDY和对应梯度值
        DX, DY, W = [], [], []

        # 设置初值
        ans = [0] * self.BinNum

        # 图像梯度直方图统计的像素范围
        k = 0
        for i in range(-kp.r, kp.r + 1):
            xx = x + i
            if xx <= 0 or xx >= img.shape[0] - 1:
                continue
            for j in range(-kp.r, kp.r + 1):
                yy = y + j
                if yy <= 0 or yy >= img.shape[1] - 1:
                    continue

                dx = img[xx][yy + 1] - img[xx][yy - 1]
                dy = img[xx - 1][yy] - img[xx + 1][yy]

                DX.append(dx)
                DY.append(dy)
                W.append((i * i + j * j) * exp_scale)

        # 方向 倒数 权重
        W = np.exp(np.array(W))
        DX, DY = np.array(DX), np.array(DY)
        Ori = np.arctan2(DY, DX) * 180 / np.pi
        Mag = (np.array(DY) ** 2 + np.array(DX) ** 2) ** 0.5

        # 计算直方图的每个bin
        for k in range(Ori.shape[0]):
            bin = int(np.round((self.BinNum / 360.0) * Ori[k]))
            if bin >= self.BinNum:
                bin -= self.BinNum
            elif bin < 0:
                bin += self.BinNum
            ans[bin] += W[k] * Mag[k]

        # 用高斯平滑
        temp = [ans[self.BinNum - 1], ans[self.BinNum - 2], ans[0], ans[1]]
        ans.insert(0, temp[0])
        ans.insert(0, temp[1])
        ans.insert(len(ans), temp[2])
        ans.insert(len(ans), temp[3])

        # 统计直方图
        hist = []
        for i in range(self.BinNum):
            hist.append(
                (ans[i] + ans[i + 4]) * (1.0 / 16.0) + (ans[i + 1] + ans[i + 3]) * (4.0 / 16.0) +
                ans[i + 2] * (6.0 / 16.0))

        return max(hist), hist

    def Cal_KeyPoints(self):
        """
        计算关键点的信息
        :return: 关键点列表
        """
        boder = 1  # 控制边缘
        key_points = []
        for oo in range(self.o):
            for ss in range(1, self.s - 2):
                img_now, img_back, img_nex = self.DoG[oo][ss], self.DoG[oo][ss - 1], self.DoG[oo][ss + 1]
                for i in range(boder,img_now.shape[0]-boder):
                    for j in range(boder,img_now.shape[1]-boder):
                        val = img_now[i][j]
                        # 阈值筛选以及是否是极值的初步判定
                        if np.abs(val) > self.the and \
                        ((val < 0 and val <= img_now[i - 1:i + 2, j - 1:j + 2].min() and \
                                   val <= img_back[i - 1:i + 2, j - 1:j + 2].min() and val <= img_nex[i - 1:i + 2, j - 1:j + 2].min()) \
                                    or \
                                 (val > 0 and val >= img_now[i - 1:i + 2, j - 1:j + 2].max() and \
                                  val >= img_back[i - 1:i + 2, j - 1:j + 2].max() and val >= img_nex[i - 1:i + 2,j - 1:j + 2].max())):
                            # 调整并判断极值点
                            kp, x, y = self.Is_Local_Extrema(oo, ss, i, j)
                            if kp is None:
                                continue
                            # 获得主方向的极大值还有对应的直方图
                            max_D, hist = self.Get_MainDir(kp, x, y)
                            oth_D = max_D * 0.8  # 大于0.8就预设为次方向
                            for k in range(self.BinNum):
                                # 抛物线插值
                                L = (k - 1) % self.BinNum
                                R = (k + 1) % self.BinNum
                                if hist[k] > hist[L] and hist[k] > hist[R] and hist[k] >= oth_D:
                                    bin = k + 0.5 * (hist[L] - hist[R]) / (hist[L] - 2 * hist[k] + hist[R])
                                    if bin < 0:
                                        bin = self.BinNum + bin
                                    elif bin >= self.BinNum:
                                        bin = bin - self.BinNum
                                    kp.dir = (360.0 / self.BinNum) * bin
                                    key_points.append(copy.deepcopy(kp))
        return key_points

    def calcSIFTDescriptor(self, dex, d=4, n=8):
        kpt = self.KeyPoints[dex]
        scale = 1.0 / (1 << kpt.o)  # 缩放倍数
        scl = kpt.sigo  # 该特征点所在组的图像尺寸
        pt = [int(np.round(kpt.y * scale)), int(np.round(kpt.x * scale))]  # 坐标点取整
        img = self.G[kpt.o][kpt.layer]  # 该点所在的金字塔图像
        rows, cols = img.shape[0], img.shape[1]
        ori = kpt.dir

        # 初始参数
        cos_t = np.cos(ori * (np.pi / 180))  # 余弦值
        sin_t = np.sin(ori * (np.pi / 180))  # 正弦值
        bins_per_rad = n / 360.0
        exp_scale = -1.0 / (d * d * 0.5)  # 高斯加权参数
        hist_width = 3 * scl  # 小区域的边长
        R = int(np.round(hist_width * 1.4142135623730951 * (d + 1) * 0.5))  # 半径
        cos_t /= hist_width
        sin_t /= hist_width
        dst, X, Y, YBin, XBin, W = [], [], [], [], [], []
        hist = [0.0] * ((d + 2) * (d + 2) * (n + 2))

        # 遍历圆内所有点
        v = d // 2 - 0.5
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                yrot = j * sin_t + i * cos_t
                xrot = j * cos_t - i * sin_t
                ybin = yrot + v
                xbin = xrot + v
                y = pt[1] + i
                x = pt[0] + j

                if d > ybin > -1 < xbin < d and 0 < y < rows - 1 and 0 < x < cols - 1:
                    X.append(img[y, x + 1] - img[y, x - 1])
                    Y.append(img[y - 1, x] - img[y + 1, x])
                    YBin.append(ybin)
                    XBin.append(xbin)
                    W.append((xrot * xrot + yrot * yrot) * exp_scale)

        # 计算每个点的方向 梯度 权值
        length = len(W)
        Y, X = np.array(Y), np.array(X)
        Ori = np.arctan2(Y, X) * 180 / np.pi
        Mag = (X ** 2 + Y ** 2) ** 0.5
        W = np.exp(np.array(W))

        # 判断每个点的归属
        for k in range(length):
            ybin, xbin, obin = YBin[k], XBin[k], (Ori[k] - ori) * bins_per_rad
            y0, x0, o0 = int(ybin), int(xbin), int(obin)
            ybin -= y0
            xbin -= x0
            obin -= o0
            mag = Mag[k] * W[k]
            o0 = o0 + (n if o0 < 0 else -n)

            # 三线性插值
            v_r1 = mag * ybin
            v_r0 = mag - v_r1

            v_rc11 = v_r1 * xbin
            v_rc10 = v_r1 - v_rc11

            v_rc01 = v_r0 * xbin
            v_rc00 = v_r0 - v_rc01

            v_rco111 = v_rc11 * obin
            v_rco110 = v_rc11 - v_rco111

            v_rco101 = v_rc10 * obin
            v_rco100 = v_rc10 - v_rco101

            v_rco011 = v_rc01 * obin
            v_rco010 = v_rc01 - v_rco011

            v_rco001 = v_rc00 * obin
            v_rco000 = v_rc00 - v_rco001

            idx = ((y0 + 1) * (d + 2) + x0 + 1) * (n + 2) + o0
            hist[idx] += v_rco000
            hist[idx + 1] += v_rco001
            hist[idx + (n + 2)] += v_rco010
            hist[idx + (n + 3)] += v_rco011
            hist[idx + (d + 2) * (n + 2)] += v_rco100
            hist[idx + (d + 2) * (n + 2) + 1] += v_rco101
            hist[idx + (d + 3) * (n + 2)] += v_rco110
            hist[idx + (d + 3) * (n + 2) + 1] += v_rco111

        # 统计最终结果
        for i in range(d):
            for j in range(d):
                idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2)
                hist[idx] += hist[idx + n]
                hist[idx + 1] += hist[idx + n + 1]
                for k in range(n):
                    dst.append(hist[idx + k])

        # 归一化 门限值处理
        dst = np.array(dst[:d*d*n])
        thr = np.sqrt((dst**2).sum()) * 0.2
        for i in range(dst.shape[0]):
            dst[i] = min(dst[i], thr)
        nrm2 = np.sqrt((dst**2).sum())
        nb = 512 / max(nrm2, 1.19209290E-07)
        for i in range(dst.shape[0]):
            dst[i] = min(max(dst[i] * nb, 0), 255)

        return dst

    def Cal_Descriptors(self):
        descriptors = []
        for i in range(len(self.KeyPoints)):
            descriptors.append(self.calcSIFTDescriptor(i))
        return np.array(descriptors,dtype='float32')

    def Get_inArray(self):
        kp = np.array([[X.y, X.x] for X in self.KeyPoints], dtype='float32')

        return kp, self.descriptors

    def Get_inCV(self):
        kp = np.array([cv2.KeyPoint(X.y, X.x, X.r, X.dir, X.val, X.o) for X in self.KeyPoints])

        return kp, self.descriptors

    def Show_d(self):
        imgnow = copy.deepcopy(self.ori)
        if len(imgnow.shape) > 2:
            imgnow[:, :, 0] = self.ori[:, :, 2]
            imgnow[:, :, 2] = self.ori[:, :, 0]
        plt.imshow(imgnow)
        plt.axis('off')
        su = self.KeyPoints.shape[0]
        dex = np.random.choice(a=su, size=min(100, su), replace=False, p=None)
        for x in self.KeyPoints[dex]:
            dx = 2 * x.r * np.cos(x.dir / 180 * np.pi)
            dy = 2 * x.r * np.sin(x.dir / 180 * np.pi)
            plt.arrow(x.y, x.x, dy, dx, head_width=x.r, head_length=x.r, fc='blue', ec='blue')
            plt.scatter(x.y, x.x, s=2, c='r')
        plt.show()


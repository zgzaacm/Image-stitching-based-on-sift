from Sift import SIFT
import numpy as np
import copy
import matplotlib.pyplot as plt
import cv2


class Stitcher:
    def __init__(self, left_img, right_img, ratio=0.75):
        self.img1 = right_img
        self.img2 = left_img
        self.ratio = ratio
        self.s1 = SIFT(self.img1)
        self.s2 = SIFT(self.img2)
        self.kp1, self.des1 = self.s1.Get_inArray()
        self.kp2, self.des2 = self.s2.Get_inArray()
        self.H = self.Get_H()
        self.result = self.Change()

    def Show_h(self):
        bf = cv2.BFMatcher(crossCheck=True)
        matches = bf.match(self.des1, self.des2)
        matches = sorted(matches, key=lambda x: x.distance)
        kp11, _ = self.s1.Get_inCV()
        kp22, _ = self.s2.Get_inCV()
        I3 = cv2.drawMatches(self.img1, kp11, self.img2, kp22, matches[:20], None, flags=2)
        cv2.imshow('match', I3)
        cv2.waitKey(0)

    def Get_H(self):
        bf = cv2.BFMatcher()
        m = bf.knnMatch(self.des1, self.des2, 2)
        kp_all = []
        for x in m:
            if len(x) == 2 and x[0].distance < x[1].distance * self.ratio:
                kp_all.append([self.kp1[x[0].queryIdx], self.kp2[x[0].trainIdx]])
        kp_all = np.array(kp_all, dtype='float32')
        if kp_all.shape[0] > 4:
            H, _ = cv2.findHomography(kp_all[:, 0], kp_all[:, 1], cv2.RANSAC, 4)
            return H
        else:
            return None

    def Change(self):
        r1 = cv2.warpPerspective(self.img1, self.H, (self.img1.shape[1] + self.img2.shape[1], max(self.img1.shape[0], self.img2.shape[0])))
        r1[0:self.img2.shape[0], 0:self.img2.shape[1]] = self.img2
        rows, cols = np.where(r1[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1  # 去除黑色无用部分

        return r1[min_row:max_row, min_col:max_col, :]


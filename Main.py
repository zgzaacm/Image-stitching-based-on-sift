#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:洪卫

import tkinter as tk
from tkinter import filedialog, dialog
import os
import Stitcher
import cv2

window = tk.Tk()
window.title('全景拼接实验-zgzaacm')
window.geometry('400x500+550+200')

img_left = None
img_right = None
S = None


def open_right():
    global img_right
    if img_right is None:
        file_path = filedialog.askopenfilename(title=u'选择图片', initialdir=(os.path.expanduser(r'./imgs')))
        img_right = cv2.imread(file_path)
        tk.messagebox.showinfo(title='', message='读取成功')
    else:
        tk.messagebox.showwarning(title='WARNING', message='已经读过了')


def open_left():
    global img_left
    if img_left is None:
        file_path = filedialog.askopenfilename(title=u'选择图片', initialdir=(os.path.expanduser(r'./imgs')))
        img_left = cv2.imread(file_path)
        tk.messagebox.showinfo(title='', message='读取成功')
    else:
        tk.messagebox.showwarning(title='WARNING', message='已经读过了')


def sift():
    global S
    if img_left is None or img_right is None:
        tk.messagebox.showwarning(title='WARNING', message='请先读取图片')
    elif S is None:
        tk.messagebox.showinfo(title='', message='运行时间较长，\n完成后会有提醒，\n请勿进行其余操作！')
        A = Stitcher.Stitcher(img_left, img_right)
        S = A
        tk.messagebox.showinfo(title='', message='运行成功')
    else:
        tk.messagebox.showinfo(title='', message='已计算完毕')


def show_right():
    global S
    if S is None:
        tk.messagebox.showwarning(title='WARNING', message='请先进行SIFT')
    else:
        S.s2.Show_d()


def show_left():
    global S
    if S is None:
        tk.messagebox.showwarning(title='WARNING', message='请先进行SIFT')
    else:
        S.s1.Show_d()


def show_match():
    global S
    if S is None:
        tk.messagebox.showwarning(title='WARNING', message='请先进行SIFT')
    else:
        S.Show_h()


def show_result():
    global S
    if S is None:
        tk.messagebox.showwarning(title='WARNING', message='请先进行SIFT')
    else:
        cv2.imshow('result', S.result)
        cv2.waitKey(0)


def save_file():
    global S
    if S is None:
        tk.messagebox.showwarning(title='WARNING', message='请先进行SIFT')
    else:
        file_path = filedialog.asksaveasfilename(title=u'选择文件夹', initialdir=(os.path.expanduser('.')))
        cv2.imwrite(file_path, S.result)
        tk.messagebox.showinfo(title='', message='保存成功')


def clear_all():
    global S
    global img_right
    global img_left
    S = img_right = img_left = None


l = tk.Label(window, text=u'基于SIFT的图像拼接', font=('systemfixed', 17), width=30, height=2)
l.place(x=21, y=30, anchor='nw')
bt_left = tk.Button(window, text='选择左侧图片', bg='white', font=('systemfixed', 14), command=open_left)
bt_right = tk.Button(window, text='选择右侧图片', bg='white', font=('systemfixed', 14), command=open_right)
bt_sift = tk.Button(window, text='SIFT处理', bg='white', font=('systemfixed', 14), command=sift)
bt_showL = tk.Button(window, text='显示左图关键点', bg='white', font=('systemfixed', 14), command=show_left)
bt_showR = tk.Button(window, text='显示右图关键点', bg='white', font=('systemfixed', 14), command=show_right)
bt_show = tk.Button(window, text='显示关键点匹配', bg='white', font=('systemfixed', 14), command=show_match)
bt_show2 = tk.Button(window, text='图像拼接', bg='white', font=('systemfixed', 14), command=show_result)
bt_show3 = tk.Button(window, text='保存结果', bg='white', font=('systemfixed', 14), command=save_file)
bt_all = tk.Button(window, text='清空', bg='white', font=('systemfixed', 14), command=clear_all)
bt_show4 = tk.Button(window, text='关闭', bg='white', font=('systemfixed', 14), command=window.destroy)
bt_left.place(x=60, y=100, anchor='nw')
bt_right.place(x=200, y=100, anchor='nw')
bt_sift.place(x=150, y=150, anchor='nw')
bt_showL.place(x=40, y=200, anchor='nw')
bt_showR.place(x=200, y=200, anchor='nw')
bt_show.place(x=120, y=250, anchor='nw')
bt_show2.place(x=150, y=300, anchor='nw')
bt_show3.place(x=150, y=350, anchor='nw')
bt_all.place(x=170, y=400, anchor='nw')
bt_show4.place(x=170, y=450, anchor='nw')


window.mainloop()

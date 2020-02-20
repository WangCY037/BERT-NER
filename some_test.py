#!/usr/bin/env python
#-*- coding:utf-8 -*-

import torch

def main():
    pass


if __name__ == '__main__':
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # main()
    from sklearn.metrics import classification_report,f1_score,accuracy_score


    y_true = [0, 3, 2, 2, 2,1]
    y_pred = [0, 0, 2, 2, 1,2]

    # report=classification_report(y_true, y_pred)
    # f1=f1_score(y_true, y_pred, average='macro')
    # acc = accuracy_score(y_true , y_pred )
    # print(report)
    # print(f1,acc)

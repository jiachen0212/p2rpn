# coding=utf-8
import torch
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

'''
>>> x = torch.rand(2, 5)
            >>> x
            tensor([[ 0.3992,  0.2908,  0.9044,  0.4850,  0.6004],
                    [ 0.5735,  0.9006,  0.6797,  0.4152,  0.1732]])
            >>> torch.zeros(3, 5).scatter_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
            tensor([[ 0.3992,  0.9006,  0.6797,  0.4850,  0.6004],
                    [ 0.0000,  0.2908,  0.0000,  0.4152,  0.0000],
                    [ 0.5735,  0.0000,  0.9044,  0.0000,  0.1732]])

dim = 0， 然后后面的index是这个dim上的索引值,即按照这个位置将x放进新tensor
index的size之所以和x一致,是因为它要把每个元素对应好位置放心新tensor...   

'''


def thres_maxpooling(map2d, thre):
    h, w = map2d.size()
    allones = torch.Tensor(6, 1)     # empty tensor
    for i in range(w):
        colpartones = []
        col = list(map2d[:, i])
        siglecol = torch.zeros(6, 1)    # 存储每一列的maxval and index
        for j in range(h - 1):
            if col[j] >= thre:
                colpartones.append(col[j])  # 存好了1个1
                if col[j + 1] < thre:
                    maxval = max(colpartones)
                    id = col.index(maxval)
                    siglecol.scatter_(0, torch.tensor([[id]]), maxval)
                    # 需要清空list
                    colpartones = []
                else:  # col[j+1]>=thre    # 接着上一个1
                    if j == (h - 2) and col[-1] >= thre:
                        colpartones.append(col[-1])
                        maxval = max(colpartones)
                        id = col.index(maxval)
                        siglecol.scatter_(0, torch.tensor([[id]]), maxval)
                    elif j == (h - 2) and col[-1] < thre:
                        maxval = max(colpartones)
                        id = col.index(maxval)
                        siglecol.scatter_(0, torch.tensor([[id]]), maxval)
                    else:
                        continue  # 既然没到h-2,那就接着存1  run to line 14
            # 仅仅最后一个mask1
            else:
                if j == h - 2 and col[-1] >= thre:
                    siglecol.scatter_(0, torch.tensor([[h - 1]]), col[-1])
        # 需要使用concate拼接
        if i == 0:
            allones = siglecol.clone()
        if i > 0:
            allones = torch.cat((allones, siglecol), 1)
    return allones


thresh = 0.5
map2d = torch.rand(6, 6)
print(map2d)
start = time.time()
me_map2d = thres_maxpooling(map2d, thresh)
end = time.time()
print(start - end)

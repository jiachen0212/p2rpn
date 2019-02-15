# coding=utf-8
import torch
import time
import os
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


'''
>>> x = torch.rand(2, 5)
            >>> x
            tensor([[ 0.3992,  0.2908,  0.9044,  0.4850,  0.6004],
                    [ 0.5735,  0.9006,  0.6797,  0.4152,  0.1732]])
            >>> torch.zeros(3, 5).scatter_(0, torch.Tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
            tensor([[ 0.3992,  0.9006,  0.6797,  0.4850,  0.6004],
                    [ 0.0000,  0.2908,  0.0000,  0.4152,  0.0000],
                    [ 0.5735,  0.0000,  0.9044,  0.0000,  0.1732]])

dim = 0， 然后后面的index是这个dim上的索引值,即按照这个位置将x放进新tensor
index的size之所以和x一致,是因为它要把每个元素对应好位置放心新tensor...   

'''


def thres_maxpooling(map2d, thre):
    h, w = map2d.size()
    allones = torch.zeros(w, 1).cuda()   # empty tensor
    for i in range(w):
        colpartones = torch.zeros(h, 1).cuda()  # 用于存放可能的1序列
        col = torch.index_select(map2d, 1, torch.LongTensor([i]).cuda())  # [6, 1]
        siglecol = torch.zeros(h, 1).cuda()  # 存储每一列的maxval and index
        for j in range(h - 1):
            cur = torch.index_select(col, 0, torch.LongTensor([j]).cuda())
            later = torch.index_select(col, 0, torch.LongTensor([j + 1]).cuda())
            col_last = torch.index_select(col, 0, torch.LongTensor([h - 1]).cuda())
            if cur >= thre:
                colpartones.scatter_(0, torch.LongTensor([[j]]).cuda(), cur)   # 存好了1个1
                if later < thre:
                    maxval = torch.max(colpartones, dim=0)[0].unsqueeze(1)   # (tensor(val), tensor(index))
                    ind = torch.max(colpartones, dim=0)[1].unsqueeze(1)      # .unsqueeze(1) dim=1上扩充一下
                    siglecol.scatter_(0, ind, maxval)
                    colpartones = torch.zeros(h, 1).cuda()
                else:  # col[j+1]>=thre    # 接着上一个1
                    if j == (h - 2) and col_last >= thre:
                        colpartones.scatter_(0, torch.LongTensor([[h - 1]]).cuda(), col_last)
                        maxval = torch.max(colpartones, dim=0)[0].unsqueeze(1)
                        ind = torch.max(colpartones, dim=0)[1].unsqueeze(1)
                        siglecol.scatter_(0, ind, maxval)
                    elif j == (h - 2) and col_last < thre:
                        maxval = torch.max(colpartones, dim=0)[0].unsqueeze(1)
                        ind = torch.max(colpartones, dim=0)[1].unsqueeze(1)
                        siglecol.scatter_(0, ind, maxval)
                    else:
                        continue  # 既然没到h-2,那就接着存1  run to line 14
            # 仅仅最后一个mask1
            else:
                if j == h - 2 and col_last >= thre:
                    siglecol.scatter_(0, torch.LongTensor([[h - 1]]).cuda(), col_last)
        # 需要使用concate拼接
        if i == 0:
            allones = siglecol.clone()
        if i > 0:
            allones = torch.cat((allones, siglecol), 1)
    return allones

thresh = 0.5
map2d = (torch.rand(6, 6)).cuda()
print(map2d)
start = time.time()
me_map2d = thres_maxpooling(map2d, thresh)
end = time.time()
print(end - start)
print(me_map2d)


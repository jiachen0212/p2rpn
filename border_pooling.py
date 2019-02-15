# coding=utf-8
import torch
'''
实现, 当竖直边上出现连续的1(>thre, pos point),"maxpooling"这些1序列的分数,找到maxval和其对应的index.
'''
def thres_maxpooling(map2d, thre):
    h, w = map2d.size()
    allones = []
    for i in range(w):
        colpartones = []
        partvalid = []
        col = list(map2d[:, i])
        siglecol = []
        for j in range(h - 1):
            if col[j] >= thre:
                colpartones.append(col[j])   # 存好了1个1
                if col[j + 1] < thre:
                    maxval = max(colpartones)
                    id = col.index(maxval)
                    partvalid.append(id)
                    partvalid.append(maxval)
                    siglecol.append(partvalid)
                    # 需要清空list
                    colpartones = []
                    partvalid = []
                else:   # col[j+1]>=thre    # 接着上一个1
                    if j == (h - 2) and col[-1] >= thre:
                        colpartones.append(col[-1])
                        maxval = max(colpartones)
                        id = col.index(maxval)
                        partvalid.append(id)
                        partvalid.append(maxval)
                        siglecol.append(partvalid)
                    elif j == (h - 2) and col[-1] < thre:
                        maxval = max(colpartones)
                        id = col.index(maxval)
                        partvalid.append(id)
                        partvalid.append(maxval)
                        siglecol.append(partvalid)
                    else:
                        continue   # 既然没到h-2,那就接着存1  run to line 14
                # colpartones = []
                # partvalid = []
            # 仅仅最后一个mask1
            else:
                if j == h - 2 and col[-1] >= thre:
                    partvalid.append(h - 1)
                    partvalid.append(col[-1])
                    siglecol.append(partvalid)
        allones.append(siglecol)
    return allones


gt = torch.tensor([(1, 1, 1, 6), (3, 1, 3, 6), (3, 4, 3, 6), (5, 4, 5, 6)])
gt = list(gt)
thresh = 0.5
map2d = torch.rand(6, 6)
# map2d = torch.tensor([[ 0.7396,  0.4509,  0.8909,  0.3681,  0.7890,  0.9567],
#         [ 0.5434,  0.0961,  0.1334,  0.2063,  0.2465,  0.6216],
#         [ 0.7084,  0.9867,  0.2795,  0.7978,  0.0848,  0.4928],
#         [ 0.3604,  0.6455,  0.6454,  0.7793,  0.5924,  0.9412],
#         [ 0.5203,  0.0395,  0.2632,  0.4292,  0.7812,  0.4635],
#         [ 0.6536,  0.0871,  0.7577,  0.2555,  0.5875,  0.3833]])
print(map2d)
me_map2d = thres_maxpooling(map2d, thresh)
print(me_map2d)





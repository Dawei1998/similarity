
from sklearn import metrics

A = [0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0]
B = [0,0,2,0,1,2,1,0,0,0,1,0,0,1,1,0]
result_NMI=metrics.normalized_mutual_info_score(A, B)
print("NMI:",result_NMI)

import pandas as pd
s1=pd.Series(A) #转为series类型
s2=pd.Series(B)
lll=s1.corr(s2) #计算相关系数
print('相关系数：',lll)

import numpy as np
from collections import Counter
def entropy(D):
    count_array=np.array(list(Counter(D).values()))
    P=count_array/count_array.sum()
    H=np.dot(-P,np.log2(P))
    return H
#H（D/A）
def condition_entropy(D,A):
    A=np.array(A)
    D=np.array(D)
    H_da=0
    for i in np.unique(A):
        index_i=np.ravel(np.argwhere(A==i))
        Di=D[index_i]
        H_Di=entropy(Di)
        pi=float(Di.size)/D.size
        H_da=H_da+pi*H_Di
    return H_da
#s1条件下s2的熵
def getCondEntropy(s1 , s2):
    d = dict()
    for i in list(range(len(s1))):
        d[s1[i]] = d.get(s1[i] , []) + [s2[i]]
    return sum([entropy(d[k]) * len(d[k]) / float(len(s1)) for k in d])


print(entropy(A))
print(condition_entropy(A,B))
print(getCondEntropy(B,A))


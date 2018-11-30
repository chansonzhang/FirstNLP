# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 11/28/2018
import  numpy as np
import torch
import torch.nn as nn

a = torch.randn(18000,50)
print(a.size())
b= torch.randn(18000,1)
print(b.size())
c=torch.cat((a,b),-1)
print(c.size())
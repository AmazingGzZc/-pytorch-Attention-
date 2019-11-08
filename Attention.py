import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class Attn(nn.Module):
    def __init__(self,method,hidden_size):
        super(Attn,self).__init__()
        self.method=method
        if self.method not in ['dot','general','concat']:
            raise ValueError(self.method,'不支持。')
        self.hidden_size=hidden_size
        if self.method == 'general':
            self.attn=nn.Linear(self.hidden_size,hidden_size)
        elif self.method=='concat':
            self.attn=nn.Linear(self.hidden_size,hidden_size)
            self.v=nn.Parameter(torch.FloatTensor(hidden_size))
    def dot_score(self,hidden,encoder_output):
        return torch.sum(hidden*encoder_output,dim=2)
    def general_score(self,hidden,encoder_output):
        #print('-----',encoder_output.shape)
        energy=self.attn(encoder_output)
        #print(torch.sum(hidden*energy,dim=2),torch.sum(hidden*energy,dim=2).shape)3*11
        return torch.sum(hidden*energy,dim=2)
    def concat_score(self,hidden,encoder_output):
        energy=self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v*energy,dim=2)
    def forward(self,hidden,encoder_outputs):   #此处的前向传播就是为了计算不同的注意力机制得到的分数
        if self.method=='general':
            attn_energies=self.general_score(hidden,encoder_outputs)
        elif self.method=='concat':
            attn_energies=self.concat_score(hidden,encoder_outputs)
        elif self.method=='dot':
            attn_energies=self.dot_score(hidden,encoder_outputs)
        #attn_energies=attn_energies.t()#此处进行了维度的转置，但是自己的实验还未测试，还等待修改
        '''a=F.softmax(attn_energies,dim=1).unsqueeze(1)
        print(a.shape)'''
        #print(a,torch.sum(a,1))
        return F.softmax(attn_energies,dim=1).unsqueeze(1)
        
        
        #此文件代码参考https://pytorch.org/tutorials/beginner/chatbot_tutorial.html中Attention部分，三种打分方式均有说明

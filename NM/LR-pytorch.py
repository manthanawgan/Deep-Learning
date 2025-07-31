import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRegression(nn.Module):
    def __init__(self, learning_rate, num_inputs, sigma= 0.01): #sigma is the std for initializing weight
        super().__init__()
        self.hyperparameter()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros((1, 0), requires_grad=True )


    def forward(self, X):
        return torch.matmul((self.w, X) + self.b)
    
    def loss_functn(self, y_mean, y):
        loss = (y_mean - y) ** 2 / 2
        return loss
    
class SGD(nn.Module):
    def __inti__(self):
        super().__init__()
        self.hyperparameter()

    def step(self, ):
import math
from mimetypes import init
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def where(cond, x1, x2):
    return cond.float() * x1 + (1 - cond.float()) * x2


class BinLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        # size of input: [n, in_channels]
        # size of weight: [out_channels, in_channels]
        # size of bias: [out_channels]

        
        s = weight.size()
        n = s[1]
        m = weight.norm(1, dim=1, keepdim=True).div(n)
        #weight_hat = weight.sign().mul(m.expand(s))
        output = input.mm(weight.t())
        output = output * m.t().expand(output.size())
        """
        s = weight.size()
        n = s[1]
        m = weight.norm(1, dim=1, keepdim=True).div(n)
        weight_hat = weight.sign().mul(m.expand(s))
        output = input.mm(weight_hat.t())
        """
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        grad_input = grad_bias = None

        grad_weight = grad_output.t().mm(input)

        s = weight.size()
        n = s[1]
        m = weight.norm(1, dim=1, keepdim=True).div(n).expand(s)
        #print(m.shape, m)
        m[weight.lt(-1.0)] = 0
        m[weight.gt(1.0)] = 0
        m = m.mul(grad_weight)

        m_add = weight.sign().mul(grad_weight)
        m_add = m_add.sum(dim=1, keepdim=True).expand(s)
        m_add = m_add.mul(weight.sign()).div(n)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight.sign())
        if ctx.needs_input_grad[1]:
            grad_weight = m.add(m_add)
            #grad_weight[weight.lt(-1.0)] = 0
            #grad_weight[weight.gt(1.0)] = 0
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias


class BinActive(torch.autograd.Function):

    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        input = input.sign()
        return input

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


BinLinearFun = BinLinearFunction.apply


class XNOR_Net_BinLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(XNOR_Net_BinLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        #gain = nn.init.calculate_gain("relu")
        #nn.init.xavier_uniform_(self.weight, gain=gain)
        #nn.init.xavier_normal_(self.weight, gain=gain)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #self.weight.data.normal_(0, 1 * (math.sqrt(1. / self.in_features)))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        input = BinActive.apply(input)
        return BinLinearFun(input, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class XNOR_case1_BinLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, initial = False):
        super(XNOR_case1_BinLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial = initial
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features)* 0.01, requires_grad=True)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.scale = torch.nn.Parameter(torch.rand(1, out_features)* 0.1, requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        #gain = nn.init.calculate_gain("relu")
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #nn.init.xavier_normal_(self.weight, gain=gain)
        nn.init.kaiming_uniform_(self.scale, a=math.sqrt(5))
        #nn.init.xavier_normal_(self.scale, gain=gain)
        #self.weight.data.normal_(0, 1 * (math.sqrt(1. / self.in_features)))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        if self.initial == False :
            input = BinActive.apply(input)
            weight = BinActive.apply(self.weight)
            y = F.linear(input, weight, self.bias)
            y = y * self.scale.expand(y.size())
        else :
            y = F.linear(input, self.weight, self.bias)
        return y

################################################################
# High Capacity Expert Linear for SIGN

class BinarySoftActivation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return (input == input.max(dim=1, keepdim=True)
                [0]).view_as(input).type_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        #input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        #grad_input.masked_fill_(input.ge(1) | input.le(-1), 0)
        return grad_input


class HCE_Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, expert_num = 2, only_front = False, activation=torch.sigmoid, binarize = False):
        super(HCE_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.expert_num = expert_num
        self.binarize = binarize
        self.weight = torch.nn.Parameter(torch.Tensor(expert_num, in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(expert_num, out_features))
        else:
            self.register_parameter('bias', None)
        self.projection = nn.Linear(in_features, expert_num)
        self.activation = activation
        if binarize :
            self.scale = torch.nn.Parameter(torch.Tensor(1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.expert_num):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        if self.bias is not None:
            fan_in = np.prod(self.weight.shape[2:])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias[i], a=-bound, b=bound)
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.projection.weight, gain=gain)
        nn.init.zeros_(self.projection.bias)
        if self.binarize :
            nn.init.kaiming_uniform_(self.scale, a=math.sqrt(5))

    def forward(self, x):
        batch_size, in_feature = x.size()

        gate_x = self.activation(self.projection(x))
        gate_x = BinarySoftActivation.apply(gate_x)

        base_weight = self.weight
        weight = torch.matmul(
            gate_x,
            base_weight.view(self.expert_num, -1)
        ).view(batch_size, self.in_features, self.out_features)

        bias = None
        if self.bias is not None:
            bias = torch.matmul(gate_x, self.bias)

        if self.binarize :
            x = BinActive.apply(x)
            weight = BinActive.apply(weight)
            x = torch.unsqueeze(x, dim = 1)
            y = torch.bmm(x, weight)
            y = torch.squeeze(y, dim = 1)
            y = y * self.scale.expand(y.size())
        else :
            x = torch.unsqueeze(x, dim = 1)
            y = torch.bmm(x, weight)
            y = torch.squeeze(y, dim = 1)

        return y

################################################################
# BinarySoftActivation for SAGN

class BinarySoftActivation_Head(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return (input == input.max(dim=2, keepdim=True)
                [0]).view_as(input).type_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        #input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        #grad_input.masked_fill_(input.ge(1) | input.le(-1), 0)
        return grad_input

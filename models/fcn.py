# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

import torch


from .topkrelu import F_topk_relu
from .topkrelu import TopkReLU
# NN_ReLU = nn.ReLU
# F_relu = F.relu

NN_ReLU = TopkReLU
F_relu = F_topk_relu

class FCN(torch.nn.Module):
    def __init__(self, D_in=784, H=1024, num_classes=10):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(FCN, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, num_classes)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = x.view(-1, 28 * 28)
        h_relu = F_topk_relu(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return y_pred

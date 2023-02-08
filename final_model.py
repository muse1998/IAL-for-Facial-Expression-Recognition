import torch
from torch import nn
from modules import GCA_IAL,Transformer

class GenerateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_former = GCA_IAL()
        self.fc_f=nn.Linear(256*4*4, 7)



    def forward(self, x):
        ##logits are used to calculate the auxiliary
        x,logits1,logits2,logits3= self.s_former(x)

        ##features before the FC layer are used to form t-SNE
        ##temporal avg pooling
        x_FER =torch.mean(x,dim=1)
        ##get the final result
        x_f=self.fc_f(x_FER)



        return x_f,x_FER,logits1,logits2,logits3


if __name__ == '__main__':
    img = torch.randn((1, 16, 3, 112, 112))
    model = GenerateModel()
    model(img)

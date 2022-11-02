import torch
import torch.nn as nn


class Aggr(nn.Module):
    def __init__(self, input_dim, hidden_dim, obj_num):
        super(Aggr, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.obj_num = obj_num
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
        )
        self.aggr = nn.MaxPool1d(self.obj_num)

        self.planner = nn.Sequential(
            nn.Linear(self.hidden_dim, 2),
        )
    
    def forward(self, x, mask):
        x1 = self.encoder(x)  # (1, 6, 20)
        global_feature = self.aggr(x1.permute(0, 2, 1))  # (1, 20, 1)
        global_feature = global_feature.permute(0, 2, 1)  # (1, 1, 20)
        aggr_feature = x1 + global_feature.repeat(1, self.obj_num, 1)
        plan = self.planner(aggr_feature)
        return (plan.permute(0, 2, 1) * mask).permute(0, 2, 1)


if __name__ == '__main__':
    model = Aggr(10, 20, 6)
    bodies = torch.randn((1, 6, 10))  # batch_size, obj_num, vector_length
    eli = torch.ones((1, 6))
    plan = model(bodies, eli)
    print(bodies, plan)

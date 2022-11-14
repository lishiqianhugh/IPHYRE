import torch
import torch.nn as nn


class MlpBase(nn.Module):
    def __init__(self, game_dim, action_dim, hidden_dim, obj_num):
        super(MlpBase, self).__init__()
        self.game_dim = game_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.obj_num = obj_num
        self.game_encoder = nn.Sequential(
            nn.Linear(self.game_dim, self.hidden_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
        )

        self.decision = nn.Sequential(
            nn.Linear(self.hidden_dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, game_data, action):
        x1 = self.game_encoder(game_data)
        x2 = self.action_encoder(action)
        x = 0 * x1 + x2
        out = self.decision(x)
        return out


if __name__ == '__main__':
    model = MlpBase(9, 9, 32, 6)
    game_data = torch.randn((2, 6, 9))
    action = torch.randn((2, 6, 9))  # batch_size, obj_num, vector_length
    out = model(game_data, action)
    print(game_data.shape, action.shape, out.shape)

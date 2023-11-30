import torch
import torch.nn as nn
import timm


class GlobalFusion(nn.Module):
    def __init__(self, game_dim, action_dim, hidden_dim, mode='add'):
        super(GlobalFusion, self).__init__()
        assert mode in ['add', 'cat']
        self.game_dim = game_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.game_encoder = nn.Sequential(
            nn.Linear(self.game_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.ReLU(),
        )
        if self.mode == 'cat':
            self.hidden_dim *= 2
        self.decision = nn.Sequential(
            nn.Linear(self.hidden_dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, game_data, action):
        game_data = game_data.view(game_data.shape[0], -1)
        x1 = self.game_encoder(game_data)
        x2 = self.action_encoder(action)
        if self.mode == 'add':
            x = x1 + x2
        else:
            x = torch.cat((x1, x2), -1)
        out = self.decision(x)
        return out


class ObjectFusion(nn.Module):
    def __init__(self, game_dim, action_dim, hidden_dim, obj_num, mode='add'):
        super(ObjectFusion, self).__init__()
        assert mode in ['add', 'cat']
        self.game_dim = game_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.obj_num = obj_num
        self.mode = mode
        self.game_encoder = nn.Sequential(
            nn.Linear(self.game_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.ReLU(),
        )
        if self.mode == 'cat':
            self.hidden_dim *= 2
        self.decision = nn.Sequential(
            nn.Linear(self.hidden_dim * self.obj_num, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, game_data, action):
        b, n, _ = game_data.shape
        game_data = game_data.view(-1, game_data.shape[-1])
        action = action[:, :, None]
        action = action.view(-1, action.shape[-1])
        x1 = self.game_encoder(game_data)
        x2 = self.action_encoder(action)
        x1 = x1.reshape(b, n, -1)
        x2 = x2.reshape(b, n, -1)
        if self.mode == 'add':
            x = x1 + x2
        else:
            x = torch.cat((x1, x2), -1)
        x = x.reshape(b, -1)
        out = self.decision(x)
        return out


class VisionFusion(nn.Module):
    def __init__(self, game_dim, action_dim, hidden_dim, alpha, beta, mode='add'):
        super(VisionFusion, self).__init__()
        assert mode in ['add', 'cat']
        self.game_dim = game_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.image_encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.image_encoder.head = nn.Linear(768, hidden_dim)
        self.game_encoder = nn.Sequential(
            nn.Linear(self.game_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.ReLU(),
        )
        if self.mode == 'cat':
            self.hidden_dim *= 3
        self.decision = nn.Sequential(
            nn.Linear(self.hidden_dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, game_data, action, image):
        game_data = game_data.view(game_data.shape[0], -1)
        x1 = self.action_encoder(action)
        x2 = self.game_encoder(game_data)
        visual_feature = self.image_encoder(image.cuda())
        if self.mode == 'add':
            x = x1 + self.alpha * x2 + self.beta * visual_feature
        else:
            x = torch.cat((x1, x2, visual_feature), -1)
        out = self.decision(x)
        return out


if __name__ == '__main__':
    model = GlobalFusion(9, 9, 32, 6)
    game_data = torch.randn((2, 6, 9))
    action = torch.randn((2, 6, 9))  # batch_size, obj_num, vector_length
    out = model(game_data, action)
    print(game_data.shape, action.shape, out.shape)

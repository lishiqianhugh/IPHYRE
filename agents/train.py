import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.functional import one_hot
import argparse
import logging

from dataset.IPHYRE import IPHYREData
from plan_ahead_models import MlpBase
from utils import setup_seed
from game_paras import max_eli_obj_num


def arg_parse():
    parser = argparse.ArgumentParser(description='Plan Ahead Parameters')
    parser.add_argument('--fold', required=False, type=int, help='from 1 to 4', default=1)
    parser.add_argument('--epoch', type=int, help='training epoch', default=10)
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.01)
    parser.add_argument('--save_interval', type=int, help='save after how many epochs', default=1)

    return parser.parse_args()


args = arg_parse()


def train(train_loader, model, opt, loss_fn):
    model.train()
    for i in range(args.epoch):
        sum_loss = []
        correct = 0
        for batch_idx, (initial_scenes, body_property, actions, label) in enumerate(train_loader):
            body_property, actions, label = body_property.to(device), actions.to(device), label.to(device)
            body_property = Variable(body_property, requires_grad=True)
            actions = Variable(actions, requires_grad=True)
            label = Variable(label, requires_grad=True)
            body_property = body_property.view(body_property.shape[0], -1)

            opt.zero_grad()
            out = model(body_property, actions)
            pred = torch.argmax(out, dim=-1).float()
            correct += (pred == label[:, 0]).sum().cpu().detach().numpy()
            label_one_hot = one_hot(label[:, 0].to(torch.int64), 2).float().to(device)
            loss = loss_fn(out, label_one_hot)
            loss.backward()
            opt.step()
            scheduler.step()
            sum_loss.append(loss.cpu().detach().numpy())

        mean_loss = np.mean(sum_loss)
        mean_acc = correct / len(train_loader.dataset)
        info = f"#######  epoch {i} loss : {mean_loss} acc: {mean_acc} #########"
        print(info)
        logging.info(info)


def eval(test_loader, model, loss_fn):
    model.eval()
    with torch.no_grad():
        sum_loss = []
        correct = 0
        for batch_idx, (initial_scenes, body_property, actions, label) in enumerate(test_loader):
            body_property, actions, label = body_property.to(device), actions.to(device), label.to(device)
            body_property = body_property.view(body_property.shape[0], -1)

            out = model(body_property, actions)
            pred = torch.argmax(out, dim=-1).float()
            correct += (pred == label[:, 0]).sum().cpu().detach().numpy()
            label_one_hot = one_hot(label[:, 0].to(torch.int64), 2).float().to(device)
            loss = loss_fn(out, label_one_hot)
            sum_loss.append(loss.cpu().detach().numpy())

        mean_loss = np.mean(sum_loss)
        mean_acc = correct / len(test_loader.dataset)
        info = f"#######  test loss: {mean_loss} mean acc: {mean_acc} #########"
        print(info)
        logging.info(info)


if __name__ == '__main__':
    # device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Using', device)

    # logging
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(filename=f'exp.log', level=20, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    setup_seed(0)
    train_set = IPHYREData(action_data_path='../dataset/action_data/',
                           game_data_path='../dataset/game_initial_data/',
                           action_num=50,
                           fold='compositional',
                           train=True)
    test_set = IPHYREData(action_data_path='../dataset/action_data/',
                          game_data_path='../dataset/game_initial_data/',
                          action_num=50,
                          fold='compositional',
                          train=False)
    kwargs = {'pin_memory': True, 'num_workers': 0}
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    # model
    model = MlpBase(game_dim=12*9, action_dim=6, hidden_dim=128, obj_num=max_eli_obj_num)
    model.to(device)

    # optimization
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-6)
    loss_fn = torch.nn.BCELoss()

    train(train_loader, model, opt, loss_fn)
    eval(test_loader, model, loss_fn)

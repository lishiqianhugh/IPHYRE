import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import logging
import optuna

from iphyre_data import IPHYRESeqData
from agents.plan_in_situ.offline_RL.decision_transformer import DecisionTransformer
from utils import setup_seed


def arg_parse():
    parser = argparse.ArgumentParser(description='Plan Ahead Parameters')
    parser.add_argument('--search', type=bool, help='whether searching hyperparameters', default=False)
    parser.add_argument('--fold', required=False, type=str, default='compositional',
                        choices=['basic', 'compositional', 'noisy', 'multi_ball'])
    parser.add_argument('--model', required=False, type=str, default='LSTM',
                        choices=['LSTM', 'VisionDT'])
    parser.add_argument('--seed', type=int, help='training seed', default=0)
    parser.add_argument('--epoch', type=int, help='training epoch', default=50)
    parser.add_argument('--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('--alpha', type=float, help='weight of VF game paras', default=1.)
    parser.add_argument('--beta', type=float, help='iweight of VF image', default=1.)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.001)
    parser.add_argument('--save_interval', type=int, help='save after how many epochs', default=1)

    return parser.parse_args()


args = arg_parse()


def train(train_loader, model, opt, loss_fn, scheduler):
    model.train()
    for i in range(args.epoch):
        sum_loss = []
        for batch_idx, (game_names, initial_scenes, body_property, actions, returns_to_go, time_steps, body_num) in enumerate(train_loader):
            bs, seq_len = body_property.shape[0: 2]
            body_property = body_property.reshape(bs, seq_len, -1)
            body_property, actions, returns_to_go, time_steps = \
            body_property.to(device), actions.to(device), returns_to_go.to(device), time_steps.to(device)
            body_property = Variable(body_property, requires_grad=True)
            actions = Variable(actions, requires_grad=True)
            returns_to_go = Variable(returns_to_go, requires_grad=True)
            time_steps = Variable(time_steps, requires_grad=True)

            body_property_gt, actions_gt, returns_to_go_gt = \
                    body_property.clone(), actions.clone(), returns_to_go.clone()

            opt.zero_grad()
            if args.model == 'VisionDT':
                initial_scenes.to(device)
                initial_scenes = Variable(initial_scenes, requires_grad=True)
                body_property_pred, actions_pred, returns_to_go_pred = \
                    model(body_property, actions, initial_scenes, returns_to_go, time_steps)
            else:
                body_property_pred, actions_pred, returns_to_go_pred = \
                    model(body_property, actions, returns_to_go, time_steps)
            body_property_pred = body_property_pred.reshape((bs, seq_len, 12, -1))
            body_property_gt = body_property_gt.reshape((bs, seq_len, 12, -1))
            loss = loss_fn(actions_pred[:, :70], actions_gt[:, :70])
            loss += loss_fn(body_property_pred[:, :, :, :], body_property_gt[:, :, :, :])
            loss += loss_fn(returns_to_go_pred, returns_to_go_gt)
            loss.backward()
            opt.step()
            scheduler.step()
            sum_loss.append(loss.cpu().detach().numpy())

        mean_loss = np.mean(sum_loss)
        info = f"epoch {i} loss : {mean_loss: .4f}"
        print(info)
        logging.info(info)
    for bp, bg in zip(body_property_pred[0], body_property_gt[0]):
        print(game_names[0], bp, bg)
    # for ap, ag in zip(actions_pred[0], actions_gt[0]):
    #     print(game_names[0], ap, ag)
    # for rp, rg in zip(returns_to_go_pred[0], returns_to_go_gt[0]):
    #     print(game_names[0], rp, rg)
    return mean_loss


def eval(test_loader, model, loss_fn):
    model.eval()
    with torch.no_grad():
        sum_loss = []
        for batch_idx, (game_names, initial_scenes, body_property, actions, returns_to_go, time_steps, body_num) in enumerate(test_loader):
            bs, seq_len = body_property.shape[0: 2]
            body_property = body_property.reshape(bs, seq_len, -1)
            body_property, actions, returns_to_go, time_steps = \
            body_property.to(device), actions.to(device), returns_to_go.to(device), time_steps.to(device)
            if args.model == 'VisionDT':
                initial_scenes.to(device)
                body_property_pred, actions_pred, returns_to_go_pred = \
                    model(body_property, actions, initial_scenes, returns_to_go, time_steps)
            else:
                body_property_pred, actions_pred, returns_to_go_pred = \
                    model(body_property, actions, returns_to_go, time_steps)
            
            loss = loss_fn(actions_pred, actions)
            loss += loss_fn(body_property_pred, body_property)
            loss += loss_fn(returns_to_go_pred, returns_to_go)
            sum_loss.append(loss.cpu().detach().numpy())

        mean_loss = np.mean(sum_loss)
        info = f"test loss : {mean_loss: .4f}"
        print(info)
        logging.info(info)
    return mean_loss


def objective(trial):
    lr = trial.suggest_float('lr', 0.001, 0.1)
    # model
    if args.model == 'LSTM':
        model = DecisionTransformer(game_dim=12 * 9, action_dim=12, hidden_dim=256, mode='cat')
    else:
        ValueError(f'No such model {args.model}')

    model.to(device)
    # optimization
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch, eta_min=1e-6)
    loss_fn = torch.nn.MSELoss()

    train(train_loader, model, opt, loss_fn, scheduler)
    eval_acc = eval(test_loader, model, loss_fn)

    return eval_acc


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
    logging.basicConfig(filename=f'logs/DTexp.log', level=20, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logging.info(f'Fold: {args.fold} Seed: {args.seed}')

    setup_seed(args.seed)
    train_set = IPHYRESeqData(data_path='dataset/offline_data/',
                           num_succeed=50,
                           num_fail=50,
                           fold=args.fold,
                           train=True)
    test_set = IPHYRESeqData(data_path='dataset/offline_data/',
                          num_succeed=50,
                          num_fail=50,
                          fold=args.fold,
                          train=False)
    kwargs = {'pin_memory': True, 'num_workers': 0}
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    if args.search:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        print('best parmas:', study.best_trial.params,
              '\n', 'best acc:', study.best_trial.values)
    else:
        # model
        if args.model == 'LSTM':
            model = DecisionTransformer(state_dim=12 * 9, act_dim=2, hidden_size=256, max_length=150, max_ep_len=1)
        else:
            ValueError(f'No such model {args.model}')

        model.to(device)
        # optimization
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch, eta_min=1e-6)
        loss_fn = torch.nn.MSELoss()

        train(train_loader, model, opt, loss_fn, scheduler)
        eval_acc = eval(test_loader, model, loss_fn)

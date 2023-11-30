import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.functional import one_hot
import argparse
import logging
import optuna

from iphyre_data import IPHYREData
from agents.fusion_models import *
from utils import setup_seed
from iphyre.games import MAX_OBJ_NUM


def arg_parse():
    parser = argparse.ArgumentParser(description='Plan Ahead Parameters')
    parser.add_argument('--search', type=bool, help='whether searching hyperparameters', default=False)
    parser.add_argument('--model', required=False, type=str, default='VisionFusion',
                        choices=['GlobalFusion', 'ObjectFusion', 'VisionFusion'])
    parser.add_argument('--mode', required=False, type=str, default='add',
                        choices=['add', 'cat'])
    parser.add_argument('--seed', type=int, help='training seed', default=0)
    parser.add_argument('--epoch', type=int, help='training epoch', default=200)
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--alpha', type=float, help='weight of VF game paras', default=0.8)
    parser.add_argument('--beta', type=float, help='weight of VF image', default=0.2)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.001)
    parser.add_argument('--save_interval', type=int, help='save after how many epochs', default=1)

    return parser.parse_args()


args = arg_parse()


def train(train_loader, model, opt, loss_fn, scheduler):
    model.train()
    for i in range(args.epoch):
        sum_loss = []
        correct = 0
        for batch_idx, (_, initial_scenes, body_property, actions, labels) in enumerate(train_loader):
            body_property, actions, labels = body_property.to(device), actions.to(device), labels.to(device)
            body_property = Variable(body_property, requires_grad=True)
            actions = Variable(actions, requires_grad=True)
            labels = Variable(labels, requires_grad=True)

            opt.zero_grad()
            if args.model == 'VisionFusion':
                initial_scenes.to(device)
                initial_scenes = Variable(initial_scenes, requires_grad=True)
                out = model(body_property, actions, initial_scenes)
            else:
                out = model(body_property, actions)
            pred = torch.argmax(out, dim=-1).float()
            batch_correct = (pred == labels[:, 0]).cpu().detach().numpy() * 1
            correct += batch_correct.sum()
            labels_one_hot = one_hot(labels[:, 0].to(torch.int64), 2).float().to(device)
            loss = loss_fn(out, labels_one_hot)
            loss.backward()
            opt.step()
            scheduler.step()
            sum_loss.append(loss.cpu().detach().numpy())

        mean_loss = np.mean(sum_loss)
        mean_acc = correct / len(train_loader.dataset)
        info = f"epoch {i} loss : {mean_loss: .4f} acc: {mean_acc: .4f}"
        print(info)
        logging.info(info)
    return mean_acc


def eval(test_loader, model, loss_fn):
    model.eval()
    with torch.no_grad():
        sum_loss = []
        correct = 0
        correct_per_game = {}
        for key in test_set.game_names:
            correct_per_game[key] = [0, 0]
        for batch_idx, (game_names, initial_scenes, body_property, actions, labels) in enumerate(test_loader):
            body_property, actions, labels = body_property.to(device), actions.to(device), labels.to(device)
            if args.model == 'VisionFusion':
                initial_scenes.to(device)
                out = model(body_property, actions, initial_scenes)
            else:
                out = model(body_property, actions)
            pred = torch.argmax(out, dim=-1).float()
            batch_correct = (pred == labels[:, 0]).cpu().detach().numpy() * 1
            correct += batch_correct.sum()
            labels_one_hot = one_hot(labels[:, 0].to(torch.int64), 2).float().to(device)
            loss = loss_fn(out, labels_one_hot)
            sum_loss.append(loss.cpu().detach().numpy())
            for game, corr, lab in zip(game_names, batch_correct, labels[:, 0]):
                correct_per_game[game][lab.int()] += corr

        mean_loss = np.mean(sum_loss)
        mean_acc = correct / len(test_loader.dataset)
        info = f"test loss: {mean_loss:.4f} mean acc: {mean_acc:.4f}\ncorrect: {correct_per_game}"
        print(info)
        logging.info(info)
    return mean_acc


def objective(trial):
    alpha = trial.suggest_float('alpha', 0.5, 1.)
    beta = trial.suggest_float('beta', 0.5, 1.)
    lr = trial.suggest_float('lr', 0.002, 0.004)
    # model
    if args.model == 'GlobalFusion':
        model = GlobalFusion(game_dim=12 * 9, action_dim=12, hidden_dim=256, mode=args.mode)
    elif args.model == 'ObjectFusion':
        model = ObjectFusion(game_dim=9, action_dim=1, hidden_dim=64, obj_num=MAX_OBJ_NUM, mode=args.mode)
    elif args.model == 'VisionFusion':
        model = VisionFusion(game_dim=12 * 9, action_dim=12, hidden_dim=256, alpha=alpha, beta=beta, mode=args.mode)
    else:
        ValueError(f'No such model {args.model}')

    model.to(device)
    # optimization
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch, eta_min=1e-6)
    loss_fn = torch.nn.BCELoss()

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
    logging.basicConfig(filename=f'logs/exp.log', level=20, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    setup_seed(args.seed)
    train_set = IPHYREData(action_data_path='./data/action_data_s50_f50/',
                           game_data_path='./data/game_initial_data/',
                           num_succeed=50,
                           num_fail=50,
                           fold='basic',
                           train=True)

    kwargs = {'pin_memory': True, 'num_workers': 0}
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    if args.search:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        print('best params:', study.best_trial.params,
              '\n', 'best acc:', study.best_trial.values)
    else:
        # model
        if args.model == 'GlobalFusion':
            model = GlobalFusion(game_dim=12 * 9, action_dim=12, hidden_dim=256, mode=args.mode)
        elif args.model == 'ObjectFusion':
            model = ObjectFusion(game_dim=9, action_dim=1, hidden_dim=64, obj_num=MAX_OBJ_NUM, mode=args.mode)
        elif args.model == 'VisionFusion':
            model = VisionFusion(game_dim=12 * 9, action_dim=12, hidden_dim=256, alpha=args.alpha, beta=args.beta, mode=args.mode)
        else:
            ValueError(f'No such model {args.model}')

        model.to(device)
        # optimization
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.model == 'VisionFusion':
            opt = torch.optim.Adam([{'params': model.image_encoder.parameters(), 'lr': args.lr / 10},
                                    {'params': model.game_encoder.parameters(), 'lr': args.lr},
                                    {'params': model.action_encoder.parameters(), 'lr': args.lr},
                                    {'params': model.decision.parameters(), 'lr': args.lr}])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch, eta_min=1e-6)
        loss_fn = torch.nn.BCELoss()

        train(train_loader, model, opt, loss_fn, scheduler)
        for fold in ['basic', 'compositional', 'noisy', 'multi_ball']:
            print(fold)
            test_set = IPHYREData(action_data_path='./data/action_data_s50_f50/',
                                  game_data_path='./data/game_initial_data/',
                                  num_succeed=50,
                                  num_fail=50,
                                  fold=fold,
                                  train=False)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            eval_acc = eval(test_loader, model, loss_fn)

import argparse
import random
from datetime import datetime
import logging
import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from torch.optim.lr_scheduler import StepLR, MultiStepLR, OneCycleLR
import torch
from dataset import Emotic_PreDataset, Emotic_PreDataset3, Emotic_PreDataset4
from models_sw import *
from utils import *
# from thop import profile
from tensorboardX import SummaryWriter
from torchmetrics import AveragePrecision
import pickle


def get_logger(filename, verbosity=1, name=None):
    """logger function for print logs."""
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def cosine_loss(x, y):

    eps = 1e-8
    x_norm = x / (x.norm(dim=1, keepdim=True) + eps)
    y_norm = y / (y.norm(dim=1, keepdim=True) + eps)
    cos_sim = (x_norm * y_norm).sum(dim=1)

    return torch.mean(1 - cos_sim)


class VI_Trainer():
    def __init__(self, args):
        # config
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        self.device = torch.device('cuda:{}'.format(args.gpu[0]))

        # make foler
        self.model_path = args.model_path
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)

        self.log_path = os.path.join(args.model_path, 'log')
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.tblog_path = os.path.join(args.model_path, 'tblog')
        if not os.path.exists(self.tblog_path):
            os.makedirs(self.tblog_path)

        # init
        self.args = args
        self.max_grad_norm = 0.5
        self.set_random_seed(args.seed)
        self.build_model()

    def train(self):
        # init
        now = datetime.now()
        timestr = now.strftime("%Y%m%d%H%M")
        tim_str_file = os.path.join(self.log_path, timestr + '_Train.log')
        self.logger = get_logger(tim_str_file)

        self.summary = SummaryWriter(log_dir=self.tblog_path, comment='Train')

        # config
        cfg_dict = vars(self.args)
        for k in cfg_dict.keys():
            self.logger.info('{}: {}'.format(k, cfg_dict[k]))

        # build dataset
        self.build_dataloader()
        # build  label
        self.build_label()
        # build optimizer
        self.build_optimizer()
        # build loss
        self.build_loss()

        # build metric
        self.metric_func = AveragePrecision(
            num_classes=self.args.num_class, average='none').to(self.device)

        # start training
        top_epoch = None
        top_score = 0
        iter_cnt = self.start_epoch * len(self.train_loader)

        for epoch in range(self.start_epoch, self.args.epochs):
            for batch_data in self.train_loader:
                iter_cnt += 1
                if iter_cnt <= self.iteration:
                    continue

                self.train_per_iter(batch_data, iter_cnt, epoch)

                if iter_cnt % 500 == 0:
                    torch.save({'model': self.model.state_dict(), 'iter': iter_cnt, 'cur_epoch': epoch,
                               'opt': self.optimizer}, self.model_path + "/BEST_checkpoint" + str(self.device) + '.pth')
                    self.logger.info('saved model')

                self.logger.info('loss value %.5f' % (self.loss.item()))

            self.scheduler.step()

        print('Top-1 score is {}, at epoch {}. Done!!!'.format(top_score, top_epoch))

    def train_per_iter(self, batch_data, iter, epoch):

        self.model.train()
        images_context, images_body, images_head, labels_cat, coord_body, coord_head = batch_data
        images_context = images_context.to(self.device)
        labels_cat = labels_cat.to(self.device)
        mapped_image_features, target_text_features = self.model(
            images_context, labels_cat)

        self.loss = self.loss_func(mapped_image_features, target_text_features)

        self.optimizer.zero_grad()
        self.loss.backward()

        has_nan_grad = False
        for param in self.model.parameters():
            if torch.isnan(param.grad).any():
                has_nan_grad = True
                break

        if has_nan_grad:
            print("Warning: Gradient is NaN! Skipping parameter update.")
            # continue

        self.optimizer.step()

        # logger
        self.summary.add_scalar('loss', self.loss.item(), iter)
        self.summary.add_scalar(
            'lr', self.optimizer.param_groups[0]['lr'], iter)

        if iter % self.args.display_freq == 0:
            self.logger.info('epoch {}/{} training iter {}/{} with loss {}'.format(
                epoch, self.args.epochs, iter, self.args.epochs * len(self.train_loader), self.loss.item()))

    @staticmethod
    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def build_label(self):
        cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
               'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
               'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']

        self.cls2idx = {}
        self.idx2cls = {}
        for idx, emotion in enumerate(cat):
            self.cls2idx[emotion] = idx
            self.idx2cls[idx] = emotion

    def build_model(self):
        # raise NotImplementedError
        self.model = model_GWT(self.args).to(self.device)

    def build_dataloader(self, use_all_flag=True):
        # raise NotImplementedError
        self.train_dataset = Emotic_PreDataset3(
            self.args.data_path, self.args.mode.split('_')[0], use_all=use_all_flag)
        self.test_dataset = Emotic_PreDataset3(
            self.args.data_path, self.args.mode.split('_')[1], use_all=True)

        self.train_loader = DataLoader(
            self.train_dataset, self.args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        self.test_loader = DataLoader(
            self.test_dataset, self.args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    def build_optimizer(self):

        self.optimizer = optim.AdamW(
            [p for p in W.parameters()], lr=1e-6, weight_decay=0.001)
        # load checkpoint if exists
        self.iteration = 0
        self.start_epoch = 0
        if self.args.ckpt is not None:
            if os.path.exists(self.args.ckpt):
                ckpt_state_dict = torch.load(
                    self.args.ckpt, map_location=self.device)
                self.model.load_state_dict(ckpt_state_dict['model'])

                self.optimizer.load_state_dict(ckpt_state_dict['opt'])
                self.iteration = ckpt_state_dict['iter']
                self.start_epoch = ckpt_state_dict['cur_epoch']

            else:
                print(self.args.ckpt, 'not Found')
                raise FileNotFoundError

        self.scheduler = StepLR(
            self.optimizer, step_size=self.args.step, gamma=self.args.gamma)

    def build_loss(self):
        self.loss_func = cosine_loss


class VI_Trainer_W(VI_Trainer):
    def __init__(self, args):
        super().__init__(args=args)

    def build_model(self):
        self.model = VI_module_w(self.args).to(
            self.device)  # Res_Emotic_v9_GWT

    def build_dataloader(self, use_all_flag=True):
        self.train_dataset = Emotic_PreDataset4(
            self.args.data_path, self.args.mode.split('_')[0], use_all=use_all_flag)
        self.test_dataset = Emotic_PreDataset4(
            self.args.data_path, self.args.mode.split('_')[1], use_all=True)

        self.train_loader = DataLoader(
            self.train_dataset, self.args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        self.test_loader = DataLoader(
            self.test_dataset, 512, shuffle=False, num_workers=8, drop_last=False)

    def build_optimizer(self):

        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                                     betas=(self.args.beta1, self.args.beta2), weight_decay=self.args.weight_decay)

        self.iteration = 0
        self.start_epoch = 0
        if self.args.ckpt is not None:
            if os.path.exists(self.args.ckpt):
                ckpt_state_dict = torch.load(
                    self.args.ckpt, map_location=self.device)
                self.model.load_state_dict(ckpt_state_dict['model'])

                self.optimizer.load_state_dict(ckpt_state_dict['opt'])
                self.iteration = ckpt_state_dict['iter']
                self.start_epoch = ckpt_state_dict['cur_epoch']

            else:
                print(self.args.ckpt, 'not Found')
                raise FileNotFoundError

        # build schedule
        # self.scheduler = MultiStepLR(self.optimizer, milestones=[8, 12], gamma=self.args.gamma)
        if args.schedule != 'cycle':
            self.scheduler = StepLR(
                self.optimizer, step_size=self.args.step, gamma=self.args.gamma)
        else:
            self.scheduler = OneCycleLR(
                self.optimizer, max_lr=0.00007, total_steps=3, )


class BaseTrainer():
    def __init__(self, args):
        # config
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        self.device = torch.device('cuda:{}'.format(args.gpu[0]))

        # make foler
        self.model_path = args.model_path
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)

        self.log_path = os.path.join(args.model_path, 'log')
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.tblog_path = os.path.join(args.model_path, 'tblog')
        if not os.path.exists(self.tblog_path):
            os.makedirs(self.tblog_path)

        # init
        self.args = args
        self.set_random_seed(args.seed)
        self.build_model()

    def train(self):
        # init
        now = datetime.now()
        timestr = now.strftime("%Y%m%d%H%M")
        tim_str_file = os.path.join(self.log_path, timestr + '_Train.log')
        self.logger = get_logger(tim_str_file)

        self.summary = SummaryWriter(log_dir=self.tblog_path, comment='Train')

        # config
        cfg_dict = vars(self.args)
        for k in cfg_dict.keys():
            self.logger.info('{}: {}'.format(k, cfg_dict[k]))

        # build dataset
        self.build_dataloader()
        # build  label
        self.build_label()
        # build optimizer
        self.build_optimizer()
        # build loss
        self.build_loss()

        # build metric
        self.metric_func = AveragePrecision(
            num_classes=self.args.num_class, average='none').to(self.device)

        # start training
        top_epoch = None
        top_score = 0
        iter_cnt = self.start_epoch * len(self.train_loader)

        for epoch in range(self.start_epoch, self.args.epochs):
            for batch_data in self.train_loader:
                iter_cnt += 1
                if iter_cnt <= self.iteration:
                    continue

                self.train_per_iter(batch_data, iter_cnt, epoch)
                if epoch >= 3:
                    self.args.eval_freq = int(10)
                    self.args.display_freq = int(10)
                if iter_cnt % self.args.eval_freq == 0:
                    score = self.eval_per_iter(iter_cnt)

                    if top_score < score:
                        torch.save({'model': self.model.state_dict(), 'iter': iter_cnt, 'cur_epoch': epoch,
                                   'opt': self.optimizer}, self.model_path + "/BEST_checkpoint" + str(self.device) + '.pth')
                        top_score = score
                        top_epoch = epoch

                    self.logger.info('Mean AP (cls) %.5f (%.5f)' %
                                     (score, top_score))

            self.scheduler.step()

        print('Top-1 score is {}, at epoch {}. Done!!!'.format(top_score, top_epoch))

    def test(self, ckpt):
        # init
        now = datetime.now()
        timestr = now.strftime("%Y%m%d%H%M")
        tim_str_file = os.path.join(self.log_path, timestr + '_Test.log')
        self.logger = get_logger(tim_str_file)

        # build dataset
        self.build_dataloader()
        # build  label
        self.build_label()
        # build metric
        self.metric_func = AveragePrecision(
            num_classes=self.args.num_class, average='none').to(self.device)

        # load checkpoint
        ckpt_state_dict = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(ckpt_state_dict['model'])

        # start testing
        map = self.eval_per_iter(iter=None)
        self.logger.info('Mean AP: %.5f' % (map))

    def train_per_iter(self, batch_data, iter, epoch):

        self.model.train()
        images_context, images_body, images_head, labels_cat, coord_body, coord_head = batch_data 

        images_context = images_context.to(self.device)
        images_body = images_body.to(self.device)
        images_head = images_head.to(self.device)
        labels_cat = labels_cat.to(self.device)
        coord_body = coord_body.to(self.device)
        coord_head = coord_head.to(self.device)

        pred, grad_d, ret_dict = self.model(
            images_context, images_body, images_head, coord_body, coord_head, labels_cat)

        global_loss = self.loss_func(pred, labels_cat)
        # self.loss_func(ret_dict['label_head'], labels_cat) # ret['label_head'] = label_head_all
        head_loss = torch.mean(torch.tensor(ret_dict['loss_head'])).cuda(
            device=int(self.args.gpu))
        # self.loss_func(ret_dict['label_body'], labels_cat) # ret['label_body'] = label_body_all
        body_loss = torch.mean(torch.tensor(ret_dict['loss_body'])).cuda(
            device=int(self.args.gpu))
        # self.loss_func(ret_dict['label_ctx'], labels_cat) # ret['label_ctx'] = label_ctx_all
        ctx_loss = torch.mean(torch.tensor(ret_dict['loss_ctx'])).cuda(
            device=int(self.args.gpu))
        # ret['VI_cls'] = label_VI
        VI_loss = self.loss_func(ret_dict['label_VI'], labels_cat)

        loss = 0.2*(global_loss+head_loss+body_loss+ctx_loss +
                    VI_loss) + self.args.grad_d_weight * grad_d

        self.optimizer.zero_grad()
        loss.backward()
        if self.args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip)
        self.optimizer.step()

        # logger
        self.summary.add_scalar('loss', loss.item(), iter)
        self.summary.add_scalar(
            'lr', self.optimizer.param_groups[0]['lr'], iter)

        if iter % self.args.display_freq == 0:
            self.logger.info('epoch {}/{} training iter {}/{} with loss {}'.format(
                epoch, self.args.epochs, iter, self.args.epochs * len(self.train_loader), loss.item()))

    def eval_per_iter(self, iter=None):
        self.model.eval()

        cat_preds = []
        cat_labels = []
        with torch.no_grad():
            for batch_data in self.test_loader:
                torch.cuda.empty_cache()
                images_context, images_body, images_head, labels_cat, coord_body, coord_head = batch_data
                images_context = images_context.to(self.device)
                images_body = images_body.to(self.device)
                images_head = images_head.to(self.device)
                labels_cat = labels_cat.to(self.device)
                coord_body = coord_body.to(self.device)
                coord_head = coord_head.to(self.device)

                pred, _, ret = self.model(
                    images_context, images_body, images_head, coord_body, coord_head, None)
                # np.save(f'{self.args.model_path}/hier_repres/ret_array_{iter}.npy',ret)
                keys = ret.keys()
                for key in keys:
                    try:
                        ret[key] = ret[key].clone().cpu().detach().numpy()
                    except:
                        continue
                # with open(f'{self.args.model_path}/hier_repres/ret_array_{iter}.pkl', 'wb') as fff:
                #     pickle.dump(ret, fff)
                # for key in keys:
                #     ret[key] = ret[key].to(self.device)


                if self.args.loss_type in ['multilabel', 'bce', 'focal', 'balance_focal']:
                    pred = torch.sigmoid(pred)
                else:
                    raise NotImplementedError

                cat_preds.append(pred)
                cat_labels.append(labels_cat)

            cat_preds = torch.cat(cat_preds, dim=0)
            cat_labels = torch.cat(cat_labels, dim=0)

        ap = self.metric_func(cat_preds, cat_labels)
        for i in range(self.args.num_class):
            self.logger.info('Category %16s %.5f' %
                             (self.idx2cls[i], ap[i].item()))
        map = sum(ap) / len(ap)

        if iter is not None:
            self.summary.add_scalar('map', map.item(), iter)

        # torch.cuda.empty_cache()

        return map.item()

    @staticmethod
    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def build_label(self):
        cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
               'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
               'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']

        self.cls2idx = {}
        self.idx2cls = {}
        for idx, emotion in enumerate(cat):
            self.cls2idx[emotion] = idx
            self.idx2cls[idx] = emotion

    def build_model(self):
        # raise NotImplementedError
        self.model = model_GWT(self.args).to(self.device)

    def build_dataloader(self, use_all_flag=True):
        # raise NotImplementedError
        self.train_dataset = Emotic_PreDataset3(
            self.args.data_path, self.args.mode.split('_')[0], use_all=use_all_flag)
        self.test_dataset = Emotic_PreDataset3(
            self.args.data_path, self.args.mode.split('_')[1], use_all=True)

        self.train_loader = DataLoader(
            self.train_dataset, self.args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        self.test_loader = DataLoader(
            self.test_dataset, self.args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    def build_optimizer(self):
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                                     betas=(self.args.beta1, self.args.beta2), weight_decay=self.args.weight_decay)

        # load checkpoint if exists
        self.iteration = 0
        self.start_epoch = 0
        if self.args.ckpt is not None:
            if os.path.exists(self.args.ckpt):
                ckpt_state_dict = torch.load(
                    self.args.ckpt, map_location=self.device)
                self.model.load_state_dict(ckpt_state_dict['model'])

                self.optimizer.load_state_dict(ckpt_state_dict['opt'])
                self.iteration = ckpt_state_dict['iter']
                self.start_epoch = ckpt_state_dict['cur_epoch']

            else:
                print(self.args.ckpt, 'not Found')
                raise FileNotFoundError

        self.scheduler = StepLR(
            self.optimizer, step_size=self.args.step, gamma=self.args.gamma)

    def build_loss(self):
        self.loss_func = DiscreteLoss(
            self.args.loss_type, self.args.discrete_loss_weight_type, self.device).to(self.device)


class ThreeModality(BaseTrainer):
    def __init__(self, args):
        super().__init__(args=args)

    def build_model(self):
        self.model = model_GWT(self.args).to(self.device)  

    # build_dataloader(self, use_all_flag=True)
    def build_dataloader(self, use_all_flag=True):
        self.train_dataset = Emotic_PreDataset4(
            self.args.data_path, self.args.mode.split('_')[0], use_all=use_all_flag)
        self.test_dataset = Emotic_PreDataset4(
            self.args.data_path, self.args.mode.split('_')[1], use_all=use_all_flag)

        self.train_loader = DataLoader(
            self.train_dataset, self.args.batch_size, shuffle=True, num_workers=16, drop_last=True)
        self.test_loader = DataLoader(
            self.test_dataset, 512, shuffle=False, num_workers=16, drop_last=False)

    def build_optimizer(self):

        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                                     betas=(self.args.beta1, self.args.beta2), weight_decay=self.args.weight_decay)

        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr,)
        # load checkpoint if exists
        self.iteration = 0
        self.start_epoch = 0
        if self.args.ckpt is not None:
            if os.path.exists(self.args.ckpt):
                ckpt_state_dict = torch.load(
                    self.args.ckpt, map_location=self.device)
                self.model.load_state_dict(ckpt_state_dict['model'])

                self.optimizer.load_state_dict(ckpt_state_dict['opt'])
                self.iteration = ckpt_state_dict['iter']
                self.start_epoch = ckpt_state_dict['cur_epoch']

            else:
                print(self.args.ckpt, 'not Found')
                raise FileNotFoundError

        # build schedule
        # self.scheduler = MultiStepLR(self.optimizer, milestones=[8, 12], gamma=self.args.gamma)
        if args.schedule != 'cycle':
            self.scheduler = StepLR(
                self.optimizer, step_size=self.args.step, gamma=self.args.gamma)
        else:
            self.scheduler = OneCycleLR(
                self.optimizer, max_lr=0.00007, total_steps=3, )


if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    parser.add_argument('--mode', type=str, default='train_test',
                        choices=['train_test', 'train_val'])
    parser.add_argument('--data_path', type=str, default='data/Emotic/',
                        help='Path to the preprocessed data files')
    parser.add_argument('--model_path', type=str, default='./checkpoints/',
                        help='Path to load model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint to continue training')

    parser.add_argument('--num_class', type=int, default=26)
    parser.add_argument('--num_block1', type=int, default=1)
    parser.add_argument('--num_block2', type=int, default=1)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.00006)
    parser.add_argument('--inside_lr', type=float, default=0.1)
    parser.add_argument('--step', type=float, default=3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.96)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=None)

    parser.add_argument('--display_freq', type=int, default=20)
    parser.add_argument('--eval_freq', type=int, default=20)
    parser.add_argument('--loss_type', type=str, default='bce',
                        choices=['l2', 'multilabel', 'bce', 'focal', 'balance_focal'])
    parser.add_argument('--discrete_loss_weight_type', type=str, default='static',
                        choices=['dynamic', 'mean', 'static'], help='weight policy for discrete loss')
    parser.add_argument('--schedule', type=str, default='nocycle')
    parser.add_argument('--innerLoss', type=str, default='BCE')
    parser.add_argument('--grad_d_weight', type=float, default='0.1')
    
    args = parser.parse_args()
    
    engine = ThreeModality(args)
    engine.train()

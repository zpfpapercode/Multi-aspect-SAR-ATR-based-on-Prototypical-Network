# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from protonet import ProtoNet
from  parser_util import get_parser
from tqdm import tqdm
import numpy as np
import torch
import os
import os.path as osp
import time
from  torch.optim import lr_scheduler

from torch.utils.tensorboard import SummaryWriter
from thop import profile
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torch.autograd import Variable
from model_deca import vgg11_mv
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch.optim as optim
import numpy as np
from prototypical_loss import euclidean_dist
from torch.nn import functional as F
from model_deca import classifier
from dataloader_tta import mv_sar_tta
'''

'''
#hyper-para
n_batch=100
learning_rate = 0.001
alpha = 0.2
lambda_= 10
plot_flag = False
shot=10
query=15
way=10
BATCH_SIZE=32
#
txt_name ='./log_1/vgg_250_2'
root_list = './data/'
info_list = './data/train_250_extend3_lee.csv'
root_list_test = './data/'
info_list_test = './data/test_tta.csv'
model_save_path='./model_save/'

epoch_nums=200
lr_deacy=[20, 40, 60, 80, 100, 120, 140, 160, 180]
#
os.environ['CUDA_VISIBLE_DEVICES'] = "0"



def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

def init_dataset(opt):
    dataset = mv_sar(info_list, root_list, transform=None)
    n_classes = len(np.unique(dataset.y))
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset

def init_sampler(opt, labels):
    classes_per_it = opt.classes_per_it_tr
    num_samples = opt.num_support_tr + opt.num_query_tr
    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)

def init_dataloader(opt):
    dataset = init_dataset(opt)
    sampler = init_sampler(opt, dataset.y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader

def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = vgg11_mv()
    model=model.cuda()
    return model
def init_classifier():
    fc=classifier(3584,10)
    fc=fc.cuda()
    return fc

def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)

def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)

def calculate_center(model):
    print("calculate_center...")
    start=time.time()
    proto_center=[[] for i in range(10)]
    # tf = transforms.Compose([ToTensor()])
    dataset = mv_sar(info_list, root_list, transform=None)
    dataloder = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    for i, sample in enumerate(dataloder):
        batch_x, batch_y = sample
        batch_x = batch_x.float()
        batch_x=batch_x.cuda()
        batch_y=batch_y.item()
        out = model(batch_x)
        proto_center[batch_y].append(out)
    tmp_list=[]

    for i in range(len(proto_center)):
        tmp_tensor=proto_center[i][0]
        for j in range(1,len(proto_center[i])):
            tmp_tensor=torch.cat((tmp_tensor,proto_center[i][j]),dim=0)
        tmp_tensor=tmp_tensor.mean(0)
        if i==0:
            prototype=tmp_tensor
        else:
            prototype=torch.cat((prototype,tmp_tensor),dim=0)
    prototype=prototype.view(len(proto_center),-1)
    print(prototype.size())
    time_end=time.time()
    print('calculate_center cost time:{:.6f}'.format(time_end - start))
    return prototype

def write_txt(): #save_train_info
    with open(txt_name, "a+") as f:
        f.writelines('20200814 eoc eca centerloss cosine-decay warmup//hyper-parameters: normal')
        f.writelines('\n')
        f.writelines('Batch Size is %d,epoch is %d'%(BATCH_SIZE,epoch_nums))
        f.writelines('\n')
        f.writelines('Learning rate:{:.3f},alpha:{:.3f},lambda1:{:d}'.format(learning_rate,alpha,lambda_))
        f.writelines('\n')
        f.writelines('lr_deacy is :')
        f.writelines(str(lr_deacy))
        f.writelines('\n')

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):

    '''
    Train the model with the prototypical learning algorithm
    '''

    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')
    fc = init_classifier()
    ce_loss = LabelSmoothingCrossEntropy()
    ce_loss=ce_loss.cuda()
    for epoch in range(opt.epochs):
        time_start=time.time()
        print('=== Epoch: {} ==='.format(epoch))
        print('training....')
        print(optim.state_dict()['param_groups'][0]['lr'])
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x,y=x.float(),y.float()
            x, y = x.cuda(), y.cuda()
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_tr)
            loss=loss.cuda()
            x_fc=fc(model_output)
            y=y.long()
            loss_ce=ce_loss(x_fc,y)
            loss+=loss+alpha*loss_ce
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Avg Train Loss: {:.6f}, Avg Train Acc: {:.6f}'.format(avg_loss, avg_acc))
        time_end=time.time()
        print('cost time:{}'.format(time_end-time_start))
        with open(txt_name, "a+") as f:
            f.writelines('Train Loss:{:.6f},Acc:{:.6f},Take Time:{:.6f}'.format(avg_loss,
                                                                                avg_acc,
                                                                                (time_end - time_start)))
            f.writelines('\n')
        lr_scheduler.step()
        if epoch%3==0:
            print("test....")
            time_start = time.time()
            model.eval()
            with torch.no_grad():
                torch.cuda.empty_cache()
                eval_acc = 0.
                conf_matrix = np.zeros((10, 10), dtype=np.int32)
                num_correct = 0
                test_num = 0
                val_iter = iter(val_dataloader)
                prototype = calculate_center(model)
                for batch in val_iter:
                    x, y = batch
                    x1 = x[:, 0, :, :, :]
                    x2 = x[:, 1, :, :, :]
                    x3 = x[:, 2, :, :, :]
                    x1 = x1.float()
                    x2 = x2.float()
                    x3 = x3.float()
                    x1, y = x1.cuda(), y.cuda()
                    x2, x3 = x2.cuda(), x3.cuda()
                    label = y
                    out1 = model(x1)
                    out2 = model(x2)
                    out3 = model(x3)
                    model_output = (out1 + out2 + out3) / 3
                    dists = euclidean_dist(model_output, prototype)
                    log_p_y = F.log_softmax(-dists, dim=1)
                    _,out_y = log_p_y.max(1)
                    test_correct = (out_y == label).sum()
                    eval_acc += test_correct.item()
                    test_num += len(label)
                    out_y = out_y.tolist()
                    label = label.tolist()
                    for i in range(len(label)):
                        conf_matrix[int(label[i])][out_y[i]] += 1
                cur_acc = float(eval_acc / test_num)
            time_end = time.time()
            print('Test Acc: {:.6f},Test Time:{:.6f}'.format(cur_acc, time_end - time_start))
            with open(txt_name,"a+") as f:
                f.writelines("epoch:%d,Test Acc:%.6f\n"%(epoch,cur_acc))
                f.write('\n')
                f.write(str(conf_matrix))
                f.write('\n')
            # if val_dataloader is None:
            #     continue
            # val_iter = iter(val_dataloader)
            # model.eval()
            # for batch in val_iter:
            #     x, y = batch
            #     x, y = x.to(device), y.to(device)
            #     model_output = model(x)
            #     loss, acc = loss_fn(model_output, target=y,
            #                         n_support=opt.num_support_val)
            #     val_loss.append(loss.item())
            #     val_acc.append(acc.item())
            # avg_loss = np.mean(val_loss[-opt.iterations:])
            # avg_acc = np.mean(val_acc[-opt.iterations:])
            # postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            #     best_acc)
            # print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            #     avg_loss, avg_acc, postfix))
            # if avg_acc >= best_acc:
            #     torch.save(model.state_dict(), best_model_path)
            #     best_acc = avg_acc
            #     best_state = model.state_dict()

    # torch.save(model.state_dict(), last_model_path)
    #
    # for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
    #     save_list_to_file(os.path.join(opt.experiment_root,
    #                                    name + '.txt'), locals()[name])

    # return best_state, best_acc, train_loss, train_acc, val_loss, val_acc

def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    # if torch.cuda.is_available() and not options.cuda:
    #     print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    tr_dataloader = init_dataloader(options)
    #another train loader to calculate prototype
    #val_dataloader to test
    # tf = transforms.Compose([ToTensor()])
    my_sar_mv_test = mv_sar_tta(info_list_test, root_list_test, transform=None)
    val_dataloader = DataLoader(my_sar_mv_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    model = init_protonet(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    train(opt=options,
          tr_dataloader=tr_dataloader,
          val_dataloader=val_dataloader,
          model=model,
          optim=optim,
          lr_scheduler=lr_scheduler)


if __name__=="__main__":
    main()
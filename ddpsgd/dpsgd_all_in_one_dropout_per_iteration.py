
# coding: utf-8

# In[ ]:


# conda install -c conda-forge mpi4py
from mpi4py import MPI
import numpy as np
from torch.optim import Optimizer
from functools import reduce
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
import json
import time
import torch
torch.manual_seed(rank * 7 + 13)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from random_dropout_generator import *
from models import *
#from utils import progress_bar
from torch.autograd import Variable
import pathlib
import time
#import random_dropout_generator
start_time = time.time()
def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)


parser = argparse.ArgumentParser(description='Decentralized SGD Test')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--algo', default="dpsgd", type=str, help='type of the algorithm, can be allreduce, extrasgd, dpsgd,central,ring_reduce,tri_reduce')
parser.add_argument('--data-distribution', default='random', type=str, help='random, or separate')
parser.add_argument('--training-batch-size', default=128, type=int, help='batch-size-for-training')
parser.add_argument('--testing-batch-size', default=100, type=int, help='batch-size-for-testing')
parser.add_argument('--cuda', action='store_true', help='use gpu')
parser.add_argument('--debug', action='store_true', help='log debugging info')
parser.add_argument('--decrease-lr',action='store_true',help='drease learning rate')
parser.add_argument('--dropout',action="store_true",help='drop switch')
args = parser.parse_args()
# logging
comm = MPI.COMM_WORLD
mode = MPI.MODE_WRONLY|MPI.MODE_CREATE|MPI.MODE_APPEND
stringargs = str(vars(args))
stringargs = "".join(x for x in stringargs if x.isalnum())
middle_name ="--lr={}".format(args.lr)+ "--decrease-lr={}".format(args.decrease_lr)+"--dropout={}".format(args.dropout) +"--algo={}  ".format(args.algo)+"--data-distribution={}  ".format(args.data_distribution)+"--cuda={}".format(args.cuda) +"--resume={}  ".format(args.resume) + "--world_size={}".format(world_size)
log_file_name = "logfile" +middle_name +".log"
# res_file to store each epoch's loss 
res_file_name = "resfile" +middle_name +".log"
fh = MPI.File.Open(comm, log_file_name, mode)
res_fh =MPI.File.Open(comm , res_file_name, mode)
fh.Set_atomicity(True)
fh.Set_atomicity(True)
def log_message(message):
    fh.Write_shared(bytes("time  {} sec :::".format(int(time.time()-start_time))+"RANK {} ::: ".format(rank) + message + "\n", 'utf8'))
def log_res(message):
    if rank==0:
        res_fh.Write_shared(bytes(message+'\n','utf8'))
if args.debug:
    log_message(">>> world size {}".format(world_size))

use_cuda = args.cuda # torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
iterations = 0
# Data
log_message('>>> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class CIFAR10OneClass(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, class_filter=(0, 1)):
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.class_filter=class_filter
        if self.train:
            new_train_data = [self.train_data[i] for i in range(len(self.train_labels)) if self.train_labels[i] in self.class_filter]
            new_train_labels = [self.train_labels[i] for i in range(len(self.train_labels)) if self.train_labels[i] in self.class_filter]
            if args.debug:
                log_message(">>> one class label {label}".format(label=new_train_labels[:20]))

            self.train_data = new_train_data
            self.train_labels = new_train_labels
        else:
            raise NotImplementedError

if args.data_distribution == 'separate':
    if world_size == 5:
        trainset = CIFAR10OneClass(root='./data', train=True, download=True, transform=transform_train, class_filter=(rank*2, rank*2+1))
    elif world_size == 10:
        trainset = CIFAR10OneClass(root='./data', train=True, download=True, transform=transform_train, class_filter=(rank))
    elif args.algo=="central" and world_size==6:
        if rank ==0:
            trainset =[]
        else:
            trainset = CIFAR10OneClass(root='./data',train=True,download=True,transform=transform_train,class_filter=((rank-1)*2,(rank-1)*2+1))   
    else:
        trainset = CIFAR10OneClass(root='./data', train=True, download=True, transform=transform_train, class_filter=((rank*2)%10,(rank*2+1)%10))
    if args.debug:
        log_message(">>> using one class {}".format(trainset.class_filter) )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.training_batch_size, shuffle=True, num_workers=0, drop_last=True)
elif args.data_distribution == 'random':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    if args.debug:
        log_message(">>> using all classes")
    sample_indices = torch.randperm(50000).tolist()
    if world_size == 5:
        sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_indices[rank*10000:(rank+1)*10000])
    elif world_size == 10:
        sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_indices[rank*5000:(rank+1)*5000])
    elif world_size == 6 and args.algo=='central':
        if rank == 0:
            sampler = torch.utils.data.sampler.SubsetRandomSampler([])
        else:
            sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_indices[((rank-1)*10000):((rank)*10000)])
        #sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_indices[(rank-1)*10000:rank*10000])
    else:
        sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_indices[rank*(50000//world_size):(rank+1)*(50000//world_size)])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.training_batch_size, sampler=sampler, num_workers=0, drop_last=True)
else:
    raise NotImplementedError



testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.testing_batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt {} .t7'.format(middle_name))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    start_time = checkpoint['start_time']
else:
    log_message('>>> Building model..')
    # net = vgg11()
    # net = torchvision.models.AlexNet(num_classes=10)
    # net = alexnet()
    net = LeNet() # you do not want to use any batchnorm in your network for extrasgd
    # net = VGG('VGG11')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

if use_cuda:
    #seperate the work load to all the gpus
    cuda_rank = rank % torch.cuda.device_count()
    net.cuda(cuda_rank)
    # open this can seperate to all gpu
    #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


def flatten_params():
    global net
    result = torch.cat([param.data.view(-1) for param in net.parameters()], 0).cpu().numpy()
    return result

def load_params(flattened):
    # if args.debug:
    #     log_message(">>> loading parameters... {}".format(flattened[-10:]))
    global net
    offset = 0
    for param in net.parameters():
        param.data.copy_(torch.from_numpy(flattened[offset:offset + param.nelement()]).view(param.size()))
        offset += param.nelement()

def allreduce():
    global net
    flatten_w = flatten_params()
    average_w = np.zeros_like(flatten_w)
    comm.Allreduce(flatten_w, average_w, MPI.SUM)
    average_w /= world_size
    load_params(average_w)
current_epoch = 0
'''
with open('dropout_5.json') as f:
   # drop_list_5 = json.load(f)
    dropout_list_5 = json.load(f)
with open('dropout_10.json') as f:
    dropout_list_10 = json.load(f)
with open('dropout_50.json') as f:
    dropout_list_50 = json.load(f)
'''    
def check_dropout_situation():
    if args.dropout == True and rank==0:
        if args.algo == 'dpsgd' or args.algo == 'extrasgd':
            random_dropout_creat_file("json"+middle_name+".json",0,world_size)
        elif args.algo == 'central':
            random_dropout_creat_file("json"+middle_name+".json",1,world_size)
        send_reqs = [comm.Isend(np.zeros(1),dest = x) for x in range(1,world_size)]
        print("rank0 waiting")
        [send_reqs[x].wait() for x in range(len(send_reqs))]
        print("rank0 finished")
if rank ==0:
    check_dropout_situation()
if args.dropout == True:
    if rank!=0:
        flag = np.zeros(1)
        comm.Recv(flag,source = 0)
        print("recved--{}".format(rank))
    with open("json"+middle_name+".json",'r') as f:
        dropout_list_list = json.load(f)
def optionsgd(neighbors_position = [-1,1],weights=[1/3,1/3]):
    global current_epoch
    global iterations
    if args.dropout == True:
        dropout_list = dropout_list_list[iterations]
        iterations += 1
    if args.debug:
        log_message(">>> dpsgd averaging...")
    if args.dropout==True:
        log_message('dropouting------------')
        if len(dropout_list) < len(neighbors_position)+1:
            return 
        if rank in dropout_list:
            flatten_w = flatten_params()
            neighbors_w = [np.zeros_like(flatten_w) for x in range(len(neighbors_position))]
            #neighbor_1_w = np.zeros_like(flatten_w)
            #neighbor_2_w = np.zeros_like(flatten_w)
            neighbors_rank = [dropout_list[(dropout_list.index(rank) + position)%len(dropout_list)] for position in neighbors_position]
            #neighbor_1 = dropout_list[current_epoch][(dropout_list[current_epoch].index(rank)-1) % len(dropout_list[current_epoch])]
            #neighbor_2 = dropout_list[current_epoch][(dropout_list[current_epoch].index(rank)+1) % len(dropout_list[current_epoch])]
            send_reqs = [comm.Isend(flatten_w,dest=neighbors_rank[index]) for index in range(len(neighbors_rank))]
            #send_req1 = comm.Isend(flatten_w, dest=neighbor_1)
            #send_req2 = comm.Isend(flatten_w, dest=neighbor_2)
            [comm.Recv(neighbors_w[index],source = neighbors_rank[index]) for index in range(len(neighbors_rank))]
            #comm.Recv(neighbor_1_w, source = neighbor_1)
            #comm.Recv(neighbor_2_w, source = neighbor_2)
            [send_reqs[index].wait() for index in range(len(neighbors_rank))]
            #send_req1.wait()
            #send_req2.wait()
            composite_weighted_neighbors_w = [neighbors_w[index]*weights[index] for index in range(len(neighbors_rank))]
            average_w = reduce(lambda x,y:x+y,composite_weighted_neighbors_w) + flatten_w*(1-reduce(lambda x,y:x+y,weights))
            load_params(average_w)
    else:
        log_message('not dropouting---')
        flatten_w = flatten_params()
        neighbors_w = [np.zeros_like(flatten_w) for x in range(len(neighbors_position))]
            #neighbor_1_w = np.zeros_like(flatten_w)
            #neighbor_2_w = np.zeros_like(flatten_w)
            ##neighbors_rank = [dropout_list[(dropout_list.index(rank) + position)%len(dropout_list)] for position in neighbor_position]
        neighbors_rank = [(rank+position)%world_size for position in neighbors_position]
            #neighbor_1 = dropout_list[current_epoch][(dropout_list[current_epoch].index(rank)-1) % len(dropout_list[current_epoch])]
            #neighbor_2 = dropout_list[current_epoch][(dropout_list[current_epoch].index(rank)+1) % len(dropout_list[current_epoch])]
        send_reqs = [comm.Isend(flatten_w,dest=neighbors_rank[index]) for index in range(len(neighbors_rank))]
            #send_req1 = comm.Isend(flatten_w, dest=neighbor_1)
            #send_req2 = comm.Isend(flatten_w, dest=neighbor_2)
        [comm.Recv(neighbors_w[index],source = neighbors_rank[index]) for index in range(len(neighbors_rank))]
            #comm.Recv(neighbor_1_w, source = neighbor_1)
            #comm.Recv(neighbor_2_w, source = neighbor_2)
        [send_reqs[index].wait() for index in range(len(neighbors_rank))]
            #send_req1.wait()
            #send_req2.wait()
        composite_weighted_neighbors_w = [neighbors_w[index]*weights[index] for index in range(len(neighbors_rank))]
        average_w = reduce(lambda x,y:x+y,composite_weighted_neighbors_w) + flatten_w*(1-reduce(lambda x,y:x+y,weights))
        load_params(average_w)
def after_update_hook():
    print(rank)
    global net
    if args.debug:
        log_message(">>> before averaging... {}".format(flatten_params()[-10:]))

    if args.algo == 'allreduce':
        if args.debug:
            log_message(">>> allreduce averaging...")
        allreduce()
    elif args.algo == 'dpsgd':
        optionsgd(neighbors_position=[-1,1],weights=[1/3,1/3])
        '''
        if args.debug:
            log_message(">>> dpsgd averaging...")
        if args.dropout==True and world_size == 5:
            log_message('dropouting------------')
            global dropout_list_5
            global current_epoch
            if len(dropout_list_5[current_epoch]) <3:
                return 
            if rank in dropout_list_5[current_epoch]:
                
                flatten_w = flatten_params()

                neighbor_1_w = np.zeros_like(flatten_w)
                neighbor_2_w = np.zeros_like(flatten_w)

                neighbor_1 = dropout_list_5[current_epoch][(dropout_list_5[current_epoch].index(rank)-1) % len(dropout_list_5[current_epoch])]
                neighbor_2 = dropout_list_5[current_epoch][(dropout_list_5[current_epoch].index(rank)+1) % len(dropout_list_5[current_epoch])]

                send_req1 = comm.Isend(flatten_w, dest=neighbor_1)
                send_req2 = comm.Isend(flatten_w, dest=neighbor_2)

                comm.Recv(neighbor_1_w, source = neighbor_1)
                comm.Recv(neighbor_2_w, source = neighbor_2)

                send_req1.wait()
                send_req2.wait()

                average_w = (neighbor_1_w + neighbor_2_w + flatten_w) / 3
                load_params(average_w)
            '''
    #elif args.algo == 'extrasgd':
        #optionsgd(neighbors_position=[-1,1],weights=[1/4,1/4])
    elif args.algo == 'ring_reduce':
        optionsgd(neighbors_position=[-1],weights=[1/2])
    elif args.algo == 'tri_reduce':
        optionsgd(neighbors_position=[-1,1,world_size//2],weights=[1/4,1/4,1/4])
    elif args.algo == 'extrasgd' and args.dropout==False:
        # 1/2 1/4 1/4
        if args.debug:
            log_message(">>> extrasgd averaging...")
        flatten_w = flatten_params()

        neighbor_1_w = np.zeros_like(flatten_w)
        neighbor_2_w = np.zeros_like(flatten_w)

        neighbor_1 = (rank-1) % world_size
        neighbor_2 = (rank+1) % world_size

        send_req1 = comm.Isend(flatten_w, dest=neighbor_1)
        send_req2 = comm.Isend(flatten_w, dest=neighbor_2)

        comm.Recv(neighbor_1_w, source = neighbor_1)
        comm.Recv(neighbor_2_w, source = neighbor_2)

        send_req1.wait()
        send_req2.wait()
        
        average_w = (neighbor_1_w + neighbor_2_w + 2 * flatten_w) / 4
        load_params(average_w)
        pass
    elif args.algo=="extrasgd" and args.dropout==True:
    	global current_epoch
    	global iterations
    	flatten_w = flatten_params()
    	dropout_list = dropout_list_list[iterations]
    	iterations+=1
    	if len(dropout_list)<3:
    		pass
    	else:
    		if rank in dropout_list:
    			neighbor_1_w = np.zeros_like(flatten_w)
    			neighbor_2_w = np.zeros_like(flatten_w)

    			neighbor_1 = dropout_list[(dropout_list.index(rank) - 1)%len(dropout_list)]
    			neighbor_2 = dropout_list[(dropout_list.index(rank) + 1)%len(dropout_list)]

    			send_req1 = comm.Isend(flatten_w, dest=neighbor_1)
    			send_req2 = comm.Isend(flatten_w, dest=neighbor_2)

    			comm.Recv(neighbor_1_w, source = neighbor_1)
    			comm.Recv(neighbor_2_w, source = neighbor_2)

    			send_req1.wait()
    			send_req2.wait()

    			average_w = (neighbor_1_w + neighbor_2_w + 2*flatten_w)/4
    			load_params(average_w)
    elif args.algo == 'central':
        global current_epoch
        #global iterations
        if args.debug:
            log_message(">>>central averaging ...")
        flatten_w = flatten_params()
        if args.dropout == True:
            dropout_list = dropout_list_list[iterations]
            iterations+=1
            if rank == 0:
                params_list = [np.zeros_like(flatten_w) for x in range(world_size)]
                [comm.Recv(params_list[x],source=x) for x in range(1,world_size) if x in dropout_list]
                sum_w = np.zeros_like(flatten_w)
                # center node does not compute gradient but all for collect params from other slave node and sum them up and average,finally resent them back the same averaged params
                for index in range(1,world_size):
                    if index in dropout_list:
                        sum_w += params_list[index]
                average_w = (sum_w)/(len(dropout_list))
                send_reqs = [comm.Isend(average_w,dest = x) for x in range(1,world_size) if x in dropout_list]
                [req.wait() for req in send_reqs]
                load_params(average_w)
            elif rank in dropout_list:
                params_w = np.zeros_like(flatten_w)
                send_req = comm.Isend(flatten_w,dest = 0)
                #send_req.wait()
                comm.Recv(params_w,source = 0)

                load_params(params_w) 
            else:
                pass
        else:
            print("central not dropout")
            if rank == 0:
                params_list = [np.zeros_like(flatten_w) for x in range(world_size)]
                print("start recv")
                [comm.Recv(params_list[x],source=x) for x in range(1,world_size)]
                print("recved rank0")
                sum_w = np.zeros_like(flatten_w)
                for m in params_list[1:]:
                    sum_w += m
                average_w = (sum_w)/(world_size - 1)
                print("start_send rank0")
                send_reqs = [comm.Isend(average_w,dest = x) for x in range(1,world_size)]
                [req.wait() for req in send_reqs]
                print("finish sending rank0")
                load_params(average_w)
            else:
                params_w = np.zeros_like(flatten_w)
                print("req_sending--rank={}".format(rank))
                send_req = comm.Isend(flatten_w,dest = 0)
                send_req.wait()
                comm.Recv(params_w,source = 0)
                print("recv --rank={}".format(rank))
                load_params(params_w)
    else:
        raise NotImplementedError

    if args.debug:
        log_message(">>> after loading... {}".format(flatten_params()[-10:]))

class DSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate

    """

    def __init__(self, params, lr, extra=False):
        defaults = dict(lr=lr, extra=extra)
        super(DSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DSGD, self).__setstate__(state)

    def step(self):
        """Performs a single optimization step.
        """
        loss = None
        global current_epoch
        for group in self.param_groups:
            extra = group['extra']
            if args.decrease_lr == True:
                lr = group['lr'] * (0.99)**(current_epoch)
            else:
                lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if extra:
                    param_state = self.state[p]
                    if 'extra_gradient_buffer' not in param_state:
                        buf_grad = param_state['extra_gradient_buffer'] = torch.zeros_like(p.data)
                        buf_grad.add_(d_p)
                        buf_weights = param_state['extra_weights_buffer'] = torch.zeros_like(p.data)
                        buf_weights.add_(p.data)
                        #p.data.add_(-group['lr'], d_p)
                        p.data.add_(-lr,d_p)
                    else:
                        buf_grad = param_state['extra_gradient_buffer']
                        buf_weights = param_state['extra_weights_buffer']
                        saved_pdata = torch.zeros_like(p.data)
                        saved_pdata.copy_(p.data)
                        #p.data.add_(-group['lr'] * (d_p - buf_grad) - buf_weights + p.data)
                        p.data.add_(-lr*(d_p - buf_grad) - buf_weights + p.data)
                        buf_weights.copy_(saved_pdata)
                        buf_grad.copy_(d_p)
                else:
                    #p.data.add_(-lr,d_p)
                    p.data.add_(-group['lr'], d_p)

        after_update_hook()
        # for group in self.param_groups:
        #     extra = group['extra']
        #     if extra:
        #         for p in group['params']:
        #             param_state = self.state[p]
        #             buf_weights = param_state['extra_weights_buffer']
        #             buf_weights.copy_(p.data)

        return loss

criterion = nn.CrossEntropyLoss()
if args.algo == 'extrasgd':
    optimizer = DSGD(net.parameters(), lr=args.lr, extra=True)
else:
    optimizer = DSGD(net.parameters(), lr=args.lr)

# Training
def train(epoch):
    global net
    log_message('>>> Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if args.algo == "central" and rank == 0:
        for batch_idx in range((50000//(world_size-1))//args.training_batch_size):
                after_update_hook()
        return 
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        ## ?? what for
        print("rank={rank_},targetlabel={labels}".format(rank_=rank,labels=targets[:10]))
        global iterations
        #if args.algo == 'central' and rank==0:
            #pass
            #after_update_hook()
            #continue
        if args.dropout ==True:
            dropout_list = dropout_list_list[iterations]
            if rank not in dropout_list:
                after_update_hook()
                continue
        #if batch_idx > (50000//world_size)//args.training_batch_size:
           # break  
        if use_cuda:
            print("using cuda")
            inputs, targets = inputs.cuda(cuda_rank), targets.cuda(cuda_rank)
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        if args.debug:
            log_message('>>> predicted {predicted}'.format(predicted=predicted.cpu().numpy()))
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        log_message('TRAIN -- epoch: {epoch} -- batch_index: {batch_idx} -- total: {total} -- loss: {loss} -- loss-avg: {loss_avg} -- accuracy: {acc}'
                    .format(epoch=epoch,
                            batch_idx = batch_idx,
                            total=len(trainloader),
                            loss = loss.data[0],
                            loss_avg= train_loss/(batch_idx+1),
                            acc=100.*correct/total))


def test(epoch):
    # only rank 0 tests
    if rank != 0:
        return
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(cuda_rank), targets.cuda(cuda_rank)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        if args.debug:
            log_message('>>> predicted {predicted}'.format(predicted=predicted.cpu().numpy()))
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        log_message('TEST -- epoch: {epoch} -- batch_index: {batch_idx} -- total: {total} -- loss: {loss} -- accuracy: {acc}'
                    .format(epoch=epoch,
                            batch_idx = batch_idx,
                            total=len(testloader),
                            loss= test_loss/(batch_idx+1),
                            acc=100.*correct/total))
    log_res('epoch:{}  '.format(epoch)+'loss:{}  '.format(test_loss/args.testing_batch_size))
        # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    
    acc = 100.*correct/total
    if epoch%50==0:
        log_message('Saving..')
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
            'start_time':start_time,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt {} .t7'.format(middle_name))
        best_acc = acc
    
for epoch in range(start_epoch,500):
    #global current_epoch
    current_epoch = epoch
    train(epoch)
    comm.Barrier()
    test(epoch)
    comm.Barrier()


fh.Sync()
fh.Close()


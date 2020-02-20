'''
This script handling the training process.
'''

import argparse
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.optim as optim
import torch.utils.data

from MY_BERT.dataset import ClassficationDataset, paired_collate_fn
from MY_BERT.model import my_model

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from itertools import chain
from MY_BERT import Constants

def cal_loss(pred, actual):
    #pred, actual 均包含首尾<cls><sep>以及pad 标签
    ''' Calculate cross entropy loss '''
    #pred, [B,seq_L,OUTSIZE]
    #actual,[B,L]
    out_size = pred.size(2)
    mask = (actual != Constants.TAG_PAD_id)  # [B, L]  # 未pad处为1  pad 处为0
    actual = actual[mask] #选出 实际需要计算loss的分类 [B*(每个seq中实际未pad个数)]
    pred = pred.masked_select(
        mask.unsqueeze(2).expand(-1, -1, out_size)
    ).contiguous().view(-1, out_size)

    assert pred.size(0) == actual.size(0)

    loss=F.cross_entropy(pred,actual)

    pred = torch.max(pred.data, 1)[1].squeeze()

    return loss,pred,actual

def cacul_accAndF1(all_pred,all_target):
    # 计算整体macro 平均值
    f1=f1_score(all_target.reshape(-1,1), all_pred.reshape(-1,1),average='macro')
    acc=accuracy_score(all_target.reshape(-1,1), all_pred.reshape(-1,1))

    return f1,acc

def train_epoch(model, data, optimizer, device):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    count = 0
    all_pred=[]
    all_target=[]


    for batch in tqdm(
            data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        src_seq,target = map(lambda x: x.to(device), batch)
        # gold = tgt_seq[:, 1:]
        # forward
        optimizer.zero_grad()
        pred = model(src_seq)


        # backward
        loss, pred,target = cal_loss(pred, target)
        loss.backward()

        #save pred and target
        pred_array=np.array(pred.cpu())

        all_pred.append(pred_array)
        all_target.append(target.cpu())

        # update parameters
        optimizer.step()

        # note keeping
        total_loss += loss.item()
        count+=target.size(0)

    # *zip all_pred and all_target
    all_pred=np.array(list(chain(*all_pred)))
    all_target=np.array(list(chain(*all_target)))

    assert len(all_pred) == len(all_target)
    #eval train
    f1,acc=cacul_accAndF1(all_pred,all_target)

    loss_per_epoch = total_loss/count

    return loss_per_epoch, f1,acc


def eval(model, data, device):
    ''' Epoch operation in training phase'''

    model.eval()

    total_loss = 0
    count = 0
    all_pred = []
    all_target = []

    with torch.no_grad():
        for batch in tqdm(
                data, mininterval=2,
                desc='  - (valid)   ', leave=False):
            # prepare data
            src_seq, target = map(lambda x: x.to(device), batch)
            # gold = tgt_seq[:, 1:]

            # forward
            pred = model(src_seq)

            # cacul_loss
            loss, pred,target = cal_loss(pred, target)

            # save pred and target
            pred_array = np.array(pred.cpu())

            all_pred.append(pred_array)
            all_target.append(target.cpu())
            # note keeping
            total_loss += loss.item()
            count += target.size(0)

    # *zip all_pred and all_target
    all_pred = np.array(list(chain(*all_pred)))
    all_target = np.array(list(chain(*all_target)))

    assert len(all_pred) == len(all_target)
    # eval train
    f1, acc = cacul_accAndF1(all_pred, all_target)

    loss_per_epoch = total_loss / count

    return loss_per_epoch, f1,acc


def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,accu,f1\n')
            log_vf.write('epoch,loss,accu,f1\n')

    valid_f1s = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_f1,train_acc = train_epoch(
            model, training_data, optimizer, device)
        print('  - (Training)   loss: {loss: 8.5f}, accu: {accu:3.3f} %, f1: {f1:3.3f} %,'\
              'elapse: {elapse:3.3f} min'.format(
                  loss=train_loss, accu=100*train_acc,f1=100*train_f1,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_f1,valid_acc = eval(model, validation_data, device)
        print('  - (Validation)   loss: {loss: 8.5f}, accu: {accu:3.3f} %, f1: {f1:3.3f} %,'\
              'elapse: {elapse:3.3f} min'.format(
                  loss=valid_loss, accu=100*valid_acc,f1=100*valid_f1,
                    elapse=(time.time()-start)/60))

        valid_f1s += [valid_f1]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_f1_{f1:3.3f}.chkpt'.format(f1=100*valid_f1)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_f1 >= max(valid_f1s):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{accu: 8.5f},{f1:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    accu=100*train_acc, f1=100*train_f1))
                log_vf.write('{epoch},{loss: 8.5f},{accu: 8.5f},{f1:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    accu=100*valid_acc, f1=100*valid_f1))
    print('    - [Info] The best valid f1 = {} .'.format(max(valid_f1s)))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default='../process_data.pth')

    parser.add_argument('-epoch', type=int, default=35)
    parser.add_argument('-batch_size', type=int, default=32)

    parser.add_argument('-out_classes', type=int, default=2)
    parser.add_argument('-d_word_vec', type=int, default=256)
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    ###################################################
    parser.add_argument('-n_layers', type=int, default=1)
    ###################################################
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default='save_model')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')

    parser.add_argument('-learning_rate', type=float, default=0.001)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    #========= Loading Dataset =========#
    data = torch.load(opt.data)
    #opt.max_token_seq_len = data['settings'].max_token_seq_len  # include the <s> and </s>  ,here just include  </s>
    opt.out_classes = data['lable_classes']
    train_loader,valid_loader,test_loader = prepare_dataloaders(data, opt)

    #========= Preparing Model =========#

    print(opt)

    device = torch.device('cuda' if opt.cuda else 'cpu')

    bert_classfication_model = my_model(opt.out_classes).to(device)

    optimizer = optim.Adam(bert_classfication_model.parameters(), lr=opt.learning_rate)


    train(bert_classfication_model, train_loader,valid_loader, optimizer, device ,opt)


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        ClassficationDataset(
            data['train']['words'],
            data['train']['lables']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)


    valid_loader = torch.utils.data.DataLoader(
        ClassficationDataset(
            data['valid']['words'],
            data['valid']['lables']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        ClassficationDataset(
            data['test']['words'],
            data['test']['lables']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    return train_loader,valid_loader,test_loader

if __name__ == '__main__':
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
  #   #[ Epoch 27 ]
  # - (Training)   loss:  0.00001, accu: 98.555 %, f1: 90.580 %,elapse: 0.575 min
  # - (Validation)   loss:  0.00001, accu: 98.741 %, f1: 94.971 %,elapse: 0.058 min
  #   - [Info] The checkpoint file has been updated.
  #  - (Training)   loss:  0.00001, accu: 98.473 %, f1: 89.663 %,elapse: 0.824 min
  # - (Validation)   loss:  0.00002, accu: 98.387 %, f1: 93.344 %,elapse: 0.078 min
  #   - [Info] The checkpoint file has been updated.
  #   - [Info] The best valid f1 = 0.9334441438051988 .

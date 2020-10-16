import torch
import torch.nn as nn
import torch.nn.functional as F

def real_loss(D_out, train_on_gpu):
    '''Calculates how close discriminator outputs are to being real.
       param, D_out: discriminator logits
       return: real loss'''
    batch_size = D_out.size(0)
    criterion = nn.BCEWithLogitsLoss()
    labels = torch.ones(batch_size)
    if train_on_gpu:
        labels = labels.cuda()
    loss = criterion(D_out.squeeze(), labels)
    
    return loss

def fake_loss(D_out, train_on_gpu):
    '''Calculates how close discriminator outputs are to being fake.
       param, D_out: discriminator logits
       return: fake loss'''
    batch_size = D_out.size(0)
    criterion = nn.BCEWithLogitsLoss()
    labels = torch.zeros(batch_size)
    if train_on_gpu:
        labels = labels.cuda()
    loss = criterion(D_out.squeeze(), labels)
    return loss
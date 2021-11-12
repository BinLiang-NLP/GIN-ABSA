# -*- coding: utf-8 -*-

import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy
import ipdb

from pytorch_pretrained_bert import BertModel
from sklearn import metrics
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset,TABSADataset

from models import GIN

import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='temp_data/'+'{0}_tokenizer.dat'.format(opt.dataset),
                step = 4 if opt.tabsa else 3)
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='temp_data/'+'{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        
        if  opt.tabsa:
            if opt.tabsa_with_absa:
                self.trainset = TABSADataset(opt.dataset_file['train'], tokenizer,True)
                self.testset = TABSADataset(opt.dataset_file['test'], tokenizer,True)
            else:
                self.trainset = TABSADataset(opt.dataset_file['train'], tokenizer,False)
                self.testset = TABSADataset(opt.dataset_file['test'], tokenizer,False)
        else:
            self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
            self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)

        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader,classifier_data_loader=None):
        max_val_acc = 0
        max_val_acc_withf1 = 0
        max_val_f1 = 0
        max_val_f1_withacc = 0

        max_val_acc_result = []
        max_val_f1_result = []

        global_step = 0
        path = None

        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            
            for i_batch, sample_batched in enumerate(train_data_loader):
                self.model.train()
                global_step += 1
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
                loss_p = criterion(outputs, targets)
                
                if self.opt.classifier:
                    outputs_classifier,cls_asp  = self.model.classifier(inputs)
                    classifier_targets= sample_batched['classifier_polarity'].to(self.opt.device)
                    loss_classifier = criterion(outputs_classifier,classifier_targets)

                if self.opt.classifier:
                    loss =  loss_p + 0.1*loss_classifier
                else:
                    loss = loss_p
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
                    val_acc, val_f1, val_result = self._evaluate_acc_f1(val_data_loader)
                    if val_acc > max_val_acc:
                        max_val_acc = val_acc
                        max_val_acc_withf1 = val_f1
                        max_val_acc_result = val_result

                    if val_f1 > max_val_f1:
                        max_val_f1 = val_f1
                        max_val_f1_withacc = val_acc
                        max_val_f1_result = val_result
        result_save_path = './result_save/sem16/'
        if not os.path.exists(result_save_path):
            os.mkdir(result_save_path)
        file_name_acc = self.opt.model_name+'_acc.npy'
        file_name_f1 = self.opt.model_name+'_f1.npy'
        numpy.save(os.path.join(result_save_path,file_name_acc),max_val_acc_result)
        numpy.save(os.path.join(result_save_path,file_name_f1),max_val_f1_result)
        logger.info('max acc:'+ str(max_val_acc))
        logger.info('max f1:'+str(max_val_f1))
        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        target_names = ['class 0', 'class 1', 'class 2']
       
        
        pre = torch.argmax(t_outputs_all, -1).cpu()

        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), average='macro')
        result = torch.argmax(t_outputs_all,-1).cpu().tolist()

        return acc, f1, numpy.array(result)

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
        if self.opt.classifier:
            classifier_data_loader = DataLoader(dataset=self.classifierset,batch_size=self.opt.batch_size,shuffle=True)

        self._reset_params()
        if self.opt.classifier:
            best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader,classifier_data_loader)
        else:
            best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        self.model.eval()

        logger.info(self.opt.model_name)

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_spc', type=str)
    parser.add_argument('--dataset', default='twitter', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=40, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=64, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=100, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--polarities_dim_classifier', default=4, type=int)
    parser.add_argument('--polarities_dim_gating', default=12, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=19, type=int, help='set seed for reproducibility')
    parser.add_argument('--classifier', action="store_true",  help='default True')
    parser.add_argument('--gating', action="store_true",  help='default True')
    parser.add_argument('--add_loss', action="store_true",  help='default True')
    parser.add_argument('--tabsa', action="store_true",  help='default True')
    parser.add_argument('--tabsa_with_absa', action="store_true",  help='default True') # if true, then use target
    parser.add_argument('--classifier_with_absa', action="store_true",  help='default True')
    parser.add_argument('--classifier_with_absa_target', action="store_true",  help='default True')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'gin': GIN,
    }
    dataset_files = {
        'sem15':{
            'train':'./datasets/15_train.seg',
            'test':'./datasets/15_test.seg',
        },
                                                            
        'sem16':{
            'train':'./datasets/16_train.seg',
            'test':'./datasets/16_test.seg',
        },        
    }
    input_colses = {
        'gin':['text_raw_indices','entity_indices', 'attribute_indices']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if not os.path.exists('log_save'):
        os.mkdir('log_save')
    log_file = 'log_save/'+'{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    if os.path.exists(log_file):
        for i in range(4):
            log_file = log_file[:-4]+'_'+str(i)+'.log'
            if not os.path.exists(log_file):
                break

    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()

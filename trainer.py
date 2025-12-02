import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, SoftmaxWeightedLoss
from utils import ContrastiveLoss
from torchvision import transforms

def trainer_Myops(args, model, snapshot_path):
    from datasets.dataset_Myops import Myops_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    db_train = Myops_dataset(base_dir=args.root_path, base_dir1=args.root_path1, base_dir2=args.root_path2, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    con_loss = ContrastiveLoss()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader) 
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        do_contrast = epoch_num > args.start_contrast_epoch
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, image1_batch, image2_batch, label_batch = sampled_batch['image'], sampled_batch['image1'], sampled_batch['image2'], sampled_batch['label']
            image_batch, image1_batch, image2_batch, label_batch = image_batch.cuda(), image1_batch.cuda(), image2_batch.cuda(), label_batch.cuda()
            out_pre, dec_seg, features_embedding_list, text_embedding_list= model(image_batch, image1_batch, image2_batch, do_contrast)
            ignores = ([2,3],[0],[0])
            loss_all = 0
            if do_contrast:
                for i in range(len(features_embedding_list)):
                    feature_list = features_embedding_list[i]
                    ignore = ignores[i]
                    loss_con = con_loss(feature_list,
                                        label_batch,
                                        text_embedding_list,
                                        ignore,
                                        sample_num = args.contrast_sample_num,
                                        )
                    loss_all += loss_con
                loss_all = loss_all /len(features_embedding_list)
            else:
                loss_all = 0
            out_cross_loss = ce_loss(out_pre, label_batch)
            out_dice_loss = dice_loss(out_pre, label_batch, softmax=True)
            out_loss = 0.2* out_cross_loss + 0.8* out_dice_loss
            dec_cross_loss = torch.zeros(1).cuda().float()
            dec_dice_loss = torch.zeros(1).cuda().float()
            for dec_pred in dec_seg:
                dec_cross_loss += ce_loss(dec_pred, label_batch)
                dec_dice_loss += dice_loss(dec_pred, label_batch, softmax=True)
            dec_loss = 0.2* dec_cross_loss + 0.8* dec_dice_loss
            
            if epoch_num < args.region_fusion_start_epoch:
                loss = out_loss * 0.0 + dec_loss+ loss_all * args.contrast_w
            else:
                loss = out_loss + 0.5 * dec_loss+ 0.5 * loss_all * args.contrast_w

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', out_dice_loss, iter_num)

            logging.info('iteration %d : loss : %f, loss_fuse: %f' % (iter_num, loss.item(), out_dice_loss.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                out_pre = torch.argmax(torch.softmax(out_pre, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', out_pre[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 20  
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 5:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
        if epoch_num >=max_epoch:    
            break



    writer.close()
    return "Training Finished!"

from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from torch.utils import data
from datasets import gf1Segmentation
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
from utils.visualizer import Visualizer
import time


def get_argparser():
    parser = argparse.ArgumentParser()

    # Save position
    parser.add_argument("--save_dir", type=str, default='./checkpoints/',
                        help="path to Dataset")

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./GF1_datasets/cropsize_321/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='gf1', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")
    parser.add_argument("--in_channels", type=int, default=4,
                        help="num input channels (default: None)")
    parser.add_argument("--feature_scale", type=int, default=2,
                        help="feature_scale (default: 2)")

    # DCNet Options
    parser.add_argument("--model", type=str, default='DCNet_L1',
                        choices=['DCNet_L1','DCNet_L12','DCNet_L123','self_contrast',
                                 'FCN','UNet','SegNet','cloudUNet','cloudSegnet'], help='model name')

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=1000000,
                        help="epoch number (default: 100k)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 4)')
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=250)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='1,4',
                        help="GPU ID")

    parser.add_argument("--continue_training", action='store_true', default=False)
    parser.add_argument("--ckpt", default=None,help="restore from checkpoint")

    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=1,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=10,
                        help="epoch interval for eval (default: 5000)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'gf1':
        train_dst = gf1Segmentation(root=opts.data_root,image_set='train_test', transform=None)
        val_dst = gf1Segmentation(root=opts.data_root,image_set='test_test', transform=None)

    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

        score = metrics.get_results()
    return score, ret_samples


def main():

    opts = get_argparser().parse_args()

    save_dir = os.path.join(opts.save_dir + opts.model + '/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Save position is %s\n'%(save_dir))

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    # select the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s,  CUDA_VISIBLE_DEVICES: %s\n" % (device, opts.gpu_id))

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)


    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=8,
                                   drop_last=True,pin_memory=False)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size, shuffle=True, num_workers=8,
                                 drop_last=True,pin_memory=False)
    print("Dataset: %s, Train set: %d, Val set: %d" % (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    model_map = {
        'self_contrast': network.self_contrast,
        'DCNet_L1': network.DCNet_L1,
        'DCNet_L12': network.DCNet_L12,
        'DCNet_L123': network.DCNet_L123,
        'FCN': network.FCN,
        'UNet': network.UNet,
        'SegNet':network.SegNet,
        'cloudSegNet':network.cloudSegNet,
        'cloudUNet':network.cloudUNet
    }

    print('Model = %s, num_classes=%d' % (opts.model, opts.num_classes))
    model = model_map[opts.model](n_classes=opts.num_classes, is_batchnorm=True, in_channels=opts.in_channels,
                                  feature_scale=opts.feature_scale, is_deconv=False)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.5)


    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss()
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s\n\n" % path)

    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint["model_state"].items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            scheduler.max_iters=opts.total_itrs
            scheduler.min_lr= opts.lr
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Continue training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        print("Best_score is %s" % (str(best_score)))
        # del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization

    interval_loss = 0
    train_loss = list()
    train_accuracy = list()
    best_val_itrs = list()
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1

        for (images, labels) in train_loader:
            if (cur_itrs)==0 or (cur_itrs) % opts.print_interval == 0:
                t1 = time.time()

            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % opts.print_interval == 0:
                interval_loss = interval_loss / opts.print_interval
                train_loss.append(interval_loss)
                t2 = time.time()
                print("Epoch %d, Itrs %d/%d, Loss=%f, Time = %f" %(cur_epochs, cur_itrs, opts.total_itrs, interval_loss,t2-t1))
                interval_loss = 0.0


            # save the ckpt file per 5000 itrs
            if (cur_itrs) % opts.val_interval == 0:
                print("validation...")
                model.eval()

                save_ckpt(save_dir + 'latest_%s_%s_itrs%s.pth' %(opts.model, opts.dataset, str(cur_itrs)))
                time_before_val = time.time()
                val_score, ret_samples = validate(opts=opts, model=model, loader=val_loader, device=device,
                                                   metrics=metrics,ret_samples_ids=vis_sample_id)

                time_after_val = time.time()
                print('Time_val = %f'%(time_after_val-time_before_val))
                print(metrics.to_str(val_score))

                train_accuracy.append(val_score['overall_acc'])
                if val_score['overall_acc'] > best_score:  # save best model
                    best_score = val_score['overall_acc']
                    save_ckpt(save_dir+'best_%s_%s_.pth' % (opts.model, opts.dataset))
                    best_val_itrs.append(cur_itrs)
                model.train()
            scheduler.step()  # update

            if cur_itrs >= opts.total_itrs:
                print(cur_itrs)
                print(opts.total_itrs)
                return

if __name__ == '__main__':
    main()

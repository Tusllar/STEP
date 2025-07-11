import argparse
import os
import numpy as np
from utils import loader, processor


import torch
import torchlight


base_path = os.path.dirname(os.path.realpath(__file__))
# data_path = os.path.join(base_path, '../../data/')

data_path = r'D:\PBL4\STEP\Data EWalk' 
ftype = ''
coords = 3
joints = 16
cycles = 1
model_path = os.path.join(base_path, 'model/features_2D_82'+ftype)


parser = argparse.ArgumentParser(description='Gait Gen')
parser.add_argument('--train', type=bool, default=True, metavar='T',
                    help='train the model (default: True)')
parser.add_argument('--smap', type=bool, default=False, metavar='S',
                    help='train the model (default: True)')
parser.add_argument('--save-features', type=bool, default=False, metavar='SF',
                    help='save penultimate layer features (default: True)')
parser.add_argument('--batch-size', type=int, default=16, metavar='B',
                    help='input batch size for training (default: 6)')
parser.add_argument('--num-worker', type=int, default=8, metavar='W',
                    help='input batch size for training (default: 4)')
parser.add_argument('--start_epoch', type=int, default=0, metavar='SE',
                    help='starting epoch of training (default: 0)')
parser.add_argument('--num_epoch', type=int, default=500, metavar='NE',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--optimizer', type=str, default='SGD', metavar='O',
                    help='optimizer (default: Adam)')
parser.add_argument('--base-lr', type=float, default=0.01, metavar='L',
                    help='base learning rate (default: 0.1)')
parser.add_argument('--step', type=list, default=[0.6, 0.75, 0.875], metavar='[S]',
                    help='fraction of steps when learning rate will be decreased (default: [0.5, 0.75, 0.875])')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='D',
                    help='Weight decay (default: 5e-4)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='EI',
                    help='interval after which model is evaluated (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='LI',
                    help='interval after which log is printed (default: 100)')
parser.add_argument('--topk', type=list, default=[1], metavar='[K]',
                    help='top K accuracy to show (default: [1])')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pavi-log', action='store_true', default=False,
                    help='pavi log')
parser.add_argument('--print-log', action='store_true', default=True,
                    help='print log')
parser.add_argument('--save-log', action='store_true', default=True,
                    help='save log')
parser.add_argument('--work-dir', type=str, default=model_path, metavar='WD',
                    help='path to save')
# TO ADD: save_result

args = parser.parse_args()
device = 'cuda:0'

test_size = 0.4
data_train, labels_train = loader.load_data_train(data_path, ftype, coords, joints,
                                              cycles=cycles, test_size=test_size)
# print('Train set size: {:d}'.format(len(data)))
# print('Test set size: {:d}'.format(len(data_test)))
data_test, labels_test = loader.load_data_test(data_path, ftype, coords, joints,
                                              cycles=cycles, test_size=test_size)
num_classes = 4
graph_dict = {'strategy': 'spatial'}
emotions = ['Angry', 'Neutral', 'Happy', 'Sad']

if args.train:
    data_loader = {
            'train': torch.utils.data.DataLoader(
                dataset=loader.TrainTestLoader(data_train, labels_train, joints, coords, num_classes),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True),
            'test': torch.utils.data.DataLoader(
                dataset=loader.TrainTestLoader(data_test, labels_test, joints, coords, num_classes),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False)}

    print('Train set size: {:d}'.format(len(data_train)))
    print('Test set size: {:d}'.format(len(data_test)))
    print('Number of classes: {:d}'.format(num_classes))
    pr = processor.Processor(args, data_loader, coords, num_classes, graph_dict, device=device,
                                verbose=False)
    pr.train()
else:
    pr = processor.Processor(args, None, coords, num_classes, graph_dict, device=device, verbose=False)
# pr.generate_confusion_matrix(ftype, data, labels, num_classes, joints, coords)

    pr.load_and_confusion_matrix(
    weights_path=r'D:\PBL4\STEP\classifier_stgcn_real_only\model_73\features_3D_82\epoch236_acc89.06_model.pth.tar',
    save_path=os.path.join('confusion_matrix.png')
    )
# labels_pred, vecs_pred = pr.generate_predictions(data, num_classes, joints, coords)
# for idx in range(labels_pred.shape[0]):
#     print('{:d}.\t{:s}'.format(idx, emotions[int(labels_pred[idx])]))
# if args.smap:
#     pr.smap()
# if args.save_features:
#     f = pr.save_best_feature(ftype, data, joints, coords)
# print('Done')

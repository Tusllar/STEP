import argparse
import os
import numpy as np
from utils import loader, processor

import torch
import torchlight


base_path = os.path.dirname(os.path.realpath(__file__))
# data_path = os.path.join(base_path, '../data')
data_path = r'D:\PBL4\STEP\data 82'
ftype = ''
coords = 3
joints = 16
cycles = 1
model_path = os.path.join(base_path, 'Model/features_2D_my_82'+ftype)


parser = argparse.ArgumentParser(description='Gait Gen')
parser.add_argument('--batch-size', type=int, default=16, metavar='B',
                    help='input batch size for training (default: 8)')
parser.add_argument('--num-worker', type=int, default=4, metavar='W',
                    help='input batch size for training (default: 4)')
parser.add_argument('--start_epoch', type=int, default=0, metavar='SE',
                    help='starting epoch of training (default: 0)')
parser.add_argument('--num_epoch', type=int, default=500, metavar='NE',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--optimizer', type=str, default='SGD', metavar='O',
                    help='optimizer (default: SGD)')
parser.add_argument('--base-lr', type=float, default=0.01, metavar='L',
                    help='base learning rate (default: 0.1)')
parser.add_argument('--step', type=list, default=[0.5, 0.75, 0.875], metavar='[S]',
                    help='fraction of steps when learning rate will be decreased (default: [0.5, 0.75, 0.875])')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='D',
                    help='Weight decay (default: 1e-4)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='EI',
                    help='interval after which model is evaluated (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='LI',
                    help='interval after which log is printed (default: 100)')
parser.add_argument('--show-topk', type=list, default=[1], metavar='[K]',
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
# Train
data_train,labels_train = loader.load_data_train(data_path, ftype, joints, coords, cycles=cycles)
# print(data_train.shape)
# print(labels_train.shape)
print('Train set size: {:d}'.format(len(data_train)))
    
# Test
data_test,labels_test= loader.load_data_test(data_path, ftype, joints, coords, cycles=cycles)
# print(data_test.shape)
print('Test set size: {:d}'.format(len(data_test)))
# print(labels_test.shape)

aff_features = len(data_train[0][0])
num_classes = np.unique(labels_train).shape[0]
data_loader = {
    'train': torch.utils.data.DataLoader(
        dataset=loader.TrainTestLoader(data_train, labels_train, joints, coords),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True),
    'test': torch.utils.data.DataLoader(
        dataset=loader.TrainTestLoader(data_test, labels_test, joints, coords),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)}
graph_dict = {'strategy': 'spatial'}
pr = processor.Processor(args, data_loader, coords, aff_features, num_classes, graph_dict, device=device)
# pr.train()

# pr = processor.Processor(args, None, coords, aff_features, num_classes, graph_dict, device=device)
# pr.load_and_confusion_matrix(
#     weights_path=r'D:\PBL4\STEP\classifier_hybrid\Model_82\features_3D_My_82\epoch73_acc100.00_model.pth.tar',
#     save_path=os.path.join('confusion_matrix.png')
# )
# pr.generate_confusion_matrix(ftype, data_test, labels_test, num_classes, joints, coords)
pr.load_and_confusion_matrix(
    weights_path=r'D:\PBL4\STEP\classifier_hybrid\Model_82\features_3D_82\epoch172_acc99.48_model.pth.tar',
    save_path=os.path.join('confusion_matrix.png')
)
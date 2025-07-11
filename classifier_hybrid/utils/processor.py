import h5py
import math
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from net import classifier
import torchlight
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix
import seaborn as sns

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Processor(object):
    """
        Processor for gait generation
    """

    def __init__(self, args, data_loader, C, F, num_classes, graph_dict, device='cuda:0'):

        self.args = args
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.train_loss = []
        self.test_loss = []
        self.accuracy = []
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.device = device
        self.io = torchlight.IO(
            self.args.work_dir,
            save_log=self.args.save_log,
            print_log=self.args.print_log)

        # model
        if not os.path.isdir(self.args.work_dir):
            os.mkdir(self.args.work_dir)
        self.model = classifier.Classifier(C, F, num_classes, graph_dict)
        self.model.cuda('cuda:0')
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        self.best_loss = math.inf
        self.step_epochs = [math.ceil(float(self.args.num_epoch * x)) for x in self.args.step]
        self.best_epoch = 0
        self.best_accuracy = np.zeros((1, np.max(self.args.show_topk)))
        self.accuracy_updated = False
        self.train_accuracy = []

        # optimizer
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError()
        self.lr = self.args.base_lr

    def adjust_lr(self):

        # if self.args.optimizer == 'SGD' and \
        if self.meta_info['epoch'] in self.step_epochs:
            lr = self.args.base_lr * (
                    0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.step_epochs)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr

    def show_epoch_info(self):

        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.args.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):

        if self.meta_info['iter'] % self.args.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.args.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def show_topk(self, k):

        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100. * sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.accuracy.append(accuracy)
        path = os.path.join(self.args.work_dir, 'accuracy.csv')
        with open(path, 'w') as f:
            f.write('epoch,accuracy\n')
            for epoch_idx, acc in enumerate(self.accuracy):
                f.write(f'{epoch_idx},{acc:.2f}\n')

        if accuracy > self.best_accuracy[0, k-1]:
            self.best_accuracy[0, k-1] = accuracy
            self.accuracy_updated = True
        else:
            self.accuracy_updated = False
        print_epoch = self.best_epoch if self.best_epoch is not None else 0
        self.io.print_log('\tTop{}: {:.2f}%. Best so far: {:.2f}% (epoch: {:d}).'.
                          format(k, accuracy, self.best_accuracy[0, k-1], print_epoch))

    def per_train(self):

        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        correct = 0
        total = 0

        for aff, gait, label in loader:
            # get data
            aff = aff.float().to(self.device)
            gait = gait.float().to(self.device)
            label = label.long().to(self.device)

            # forward
            output = self.model(aff, gait)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

            # Tính số lượng đúng cho train accuracy
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.train_loss.append(self.epoch_info['mean_loss'])
        # Ghi toàn bộ lịch sử train loss vào CSV
        path1 = os.path.join(self.args.work_dir, 'train_loss.csv')
        with open(path1, 'w') as f:
            f.write('epoch,mean_loss\n')
            for epoch_idx, loss_val in enumerate(self.train_loss):
                f.write(f'{epoch_idx},{loss_val:.6f}\n')

        # Tính và lưu train accuracy
        train_acc = 100. * correct / total if total > 0 else 0
        self.train_accuracy.append(train_acc)
        path2 = os.path.join(self.args.work_dir, 'train_accuracy.csv')
        with open(path2, 'w') as f:
            f.write('epoch,accuracy\n')
            for epoch_idx, acc in enumerate(self.train_accuracy):
                f.write(f'{epoch_idx},{acc:.2f}\n')

        self.show_epoch_info()
        self.io.print_timer()

    def per_test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for aff, gait, label in loader:

            # get data
            aff = aff.float().to(self.device)
            gait = gait.float().to(self.device)
            label = label.long().to(self.device)

            # inference
            with torch.no_grad():
                output = self.model(aff, gait)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.test_loss.append(self.epoch_info['mean_loss'])
            # Ghi toàn bộ lịch sử test loss vào CSV
            path = os.path.join(self.args.work_dir, 'test_loss.csv')
            with open(path, 'w') as f:
                f.write('epoch,mean_loss\n')
                for epoch_idx, loss_val in enumerate(self.test_loss):
                    f.write(f'{epoch_idx},{loss_val:.6f}\n')


            self.show_epoch_info()

            # show top-k accuracy
            for k in self.args.show_topk:
                self.show_topk(k)

    def train(self):

        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            self.meta_info['epoch'] = epoch

            # training
            self.io.print_log('Training epoch: {}'.format(epoch))
            self.per_train()
            self.io.print_log('Done.')

            # evaluation
            if (epoch % self.args.eval_interval == 0) or (
                    epoch + 1 == self.args.num_epoch):
                self.io.print_log('Eval epoch: {}'.format(epoch))
                self.per_test()
                self.io.print_log('Done.')

            # save model and weights
            if self.accuracy_updated:
                torch.save(self.model.state_dict(),
                           os.path.join(self.args.work_dir,
                                        'epoch{}_acc{:.2f}_model.pth.tar'.format(epoch, self.best_accuracy.item())))

        self.plot_accuracy()
        self.plot_loss1()
        self.plot_loss()


    def test(self):

        # the path of weights must be appointed
        if self.args.weights is None:
            raise ValueError('Please appoint --weights.')
        self.io.print_log('Model:   {}.'.format(self.args.model))
        self.io.print_log('Weights: {}.'.format(self.args.weights))

        # evaluation
        self.io.print_log('Evaluation Start:')
        self.per_test()
        self.io.print_log('Done.\n')

        # save the output of model
        if self.args.save_result:
            result_dict = dict(
                zip(self.data_loader['test'].dataset.sample_name,
                    self.result))
            self.io.save_pkl(result_dict, 'test_result.pkl')

    # def generate_predictions(self, data, num_classes, joints, coords):
    #     # fin = h5py.File('../data/features'+ftype+'.h5', 'r')
    #     # fkeys = fin.keys()
    #     data = np.array(data)
    #     labels_pred = np.zeros(data.shape[0])
    #     output = np.zeros((data.shape[0], num_classes))
    #     for i, each_data in enumerate(zip(data)):
    #         # get data
    #         each_data = each_data[0]
    #         each_data = np.reshape(each_data, (1, each_data.shape[0], joints, coords, 1))
    #         each_data = np.moveaxis(each_data, [1, 2, 3], [2, 3, 1])
    #         each_data = torch.from_numpy(each_data).float().to(self.device)
    #         # get label
    #         with torch.no_grad():
    #             output_torch, _ = self.model(each_data)
    #             output[i] = output_torch.detach().cpu().numpy()
    #             labels_pred[i] = np.argmax(output[i])
    #     return labels_pred, output

    def evaluate_confusion_matrix(self, weights_path, save_path=None, show_plot=True):
        """
        Load lại model từ weights_path, chạy test loader và vẽ ma trận nhầm lẫn.
        """
        # Load weights
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        all_preds = []
        all_labels = []

        # Duyệt qua toàn bộ test loader
        with torch.no_grad():
            for aff, gait, label in self.data_loader['test']:
                aff = aff.float().to(self.device)
                gait = gait.float().to(self.device)
                label = label.long().to(self.device)
                output = self.model(aff, gait)
                preds = torch.argmax(output, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(label.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Tính ma trận nhầm lẫn
        cm = confusion_matrix(all_labels, all_preds, labels=range(self.num_classes))

        # Vẽ heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show_plot:
            plt.show()
        plt.close()
        return cm
    def generate_predictions(self, data, num_classes, joints, coords):
        labels_pred = np.zeros(len(data))
        output = np.zeros((len(data), num_classes))

        for i, each_data in enumerate(data):
            # Nếu each_data là tuple (x1, x2), bạn cần chọn 1 phần tử để dự đoán.
            # Ví dụ: each_data = each_data[0]
            if isinstance(each_data, (tuple, list)):
                each_data = each_data[0]  # hoặc [1] nếu phần tử đó là đầu vào đúng

            # Reshape và xử lý cho đúng input
            each_data = np.reshape(each_data, (1, each_data.shape[0], joints, coords, 1))
            each_data = np.moveaxis(each_data, [1, 2, 3], [2, 3, 1])
            each_data = torch.from_numpy(each_data).float().to(self.device)

            # Predict
            with torch.no_grad():
                output_torch, _ = self.model(each_data)
                output[i] = output_torch.detach().cpu().numpy()
                labels_pred[i] = np.argmax(output[i])

        return labels_pred, output

    def load_best_model(self):
        # if self.best_epoch is None:
        #     self.best_epoch, best_accuracy = get_best_epoch_and_accuracy(self.args.work_dir)
        # else:
        #     best_accuracy = self.best_accuracy.item()

        filename = os.path.join((r'D:\PBL4\STEP\classifier_hybrid\Model_82\features_3D_My_82\epoch73_acc100.00_model.pth.tar'))
        self.model.load_state_dict(torch.load(filename))

    def generate_confusion_matrix(self, ftype, data, labels, num_classes, joints, coords):
        self.load_best_model()
        self.model.eval()
        labels_pred,_ = self.generate_predictions(data, num_classes, joints, coords)

        hit = np.nonzero(labels_pred == labels)
        miss = np.nonzero(labels_pred != labels)
        confusion_matrix = np.zeros((num_classes, num_classes))
        for hidx in np.arange(len(hit[0])):
            confusion_matrix[int(labels[hit[0][hidx]]), int(labels_pred[hit[0][hidx]])] += 1
        for midx in np.arange(len(miss[0])):
            confusion_matrix[int(labels[miss[0][midx]]), int(labels_pred[miss[0][midx]])] += 1
        confusion_matrix = confusion_matrix.transpose()
            # Lưu confusion matrix ra file CSV
        np.savetxt(os.path.join(self.args.work_dir, "confusion_matrix.csv"), confusion_matrix, delimiter=",", fmt='%d')
    
        # Xuất ra console
        print("Confusion Matrix:")
        print(confusion_matrix)
        # plot_confusion_matrix(confusion_matrix)

    # def load_and_confusion_matrix(self, weights_path, save_path=r'D:\PBL4\STEP\figures', show_plot=True):
    #     """
    #     Load lại model từ weights_path, chạy test và vẽ ma trận nhầm lẫn.
    #     """
    #     # Load weights
    #     self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
    #     self.model.eval()
    #     # Chạy test để lấy self.result và self.label
    #     self.per_test(evaluation=True)
    #     # Tính nhãn dự đoán
    #     pred_labels = np.argmax(self.result, axis=1)
    #     true_labels = self.label
    #     # Tính ma trận nhầm lẫn
    #     cm = confusion_matrix(true_labels, pred_labels, labels=range(self.num_classes))
    #     # Vẽ heatmap
    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    #     plt.xlabel('Predicted label')
    #     plt.ylabel('True label')
    #     plt.title('Confusion Matrix')
    #     plt.tight_layout()
    #     if save_path is not None:
    #         plt.savefig(save_path)
    #     if show_plot:
    #         plt.show()
    #     plt.close()
    #     return cm
    def load_and_confusion_matrix(self, weights_path, save_path=None, show_plot=True):
        """
        Load lại model từ weights_path, chạy test và vẽ ma trận nhầm lẫn.
        """
        # Load weights
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        # Chạy test
        self.per_test(evaluation=True)

        # Dự đoán
        pred_labels = np.argmax(self.result, axis=1)
        true_labels = self.label

        # Tính confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=range(self.num_classes))

        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        # Lưu ảnh
        # if save_path is None:
        os.makedirs(r"D:\PBL4\STEP\figures", exist_ok=True)
        # Lấy tên từ weights_path (ví dụ: hybrid_ep30.pth → hybrid_ep30.png)
        weight_name = os.path.splitext(os.path.basename(weights_path))[0]
        save_path = f"figures/confusion_matrix_{weight_name}.png"

        plt.savefig(save_path)
        print(f"[INFO] Confusion matrix saved to: {save_path}")

        if show_plot:
            plt.show()
        plt.close()

        return cm

    def plot_loss(self):
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss, label='Train Loss', color='blue')
        plt.plot(self.test_loss, label='Test Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.args.work_dir, 'loss_plot.png'))
        plt.show()

    def plot_accuracy(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.accuracy, label='Test Accuracy', color='green', marker='o', markersize=3, linestyle='-')
        plt.plot(self.train_accuracy, label='Train Accuracy', color='blue', marker='x', markersize=3, linestyle='-')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Model Accuracy Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # Lưu hình
        plt.savefig(os.path.join(self.args.work_dir, 'accuracy_plot.png'))
        plt.show()

    def plot_loss1(self):
        plt.figure(figsize=(10, 5))

        # Train loss: vẽ đường trơn
        plt.plot(self.train_loss, label='Train Loss', color='blue', marker='o', markersize=3, linestyle='-')

        # Test loss: vẽ với dấu x và scatter
        plt.plot(self.test_loss, label='Test Loss', color='orange', marker='x', linestyle='-', markersize=4)

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Lưu hình
        plt.savefig(os.path.join(self.args.work_dir, 'loss_plot1.png'))
        plt.show()
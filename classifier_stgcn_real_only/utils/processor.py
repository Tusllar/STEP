import h5py
import math
import os
import matplotlib as mpl
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import rcParams
from net import classifier
from torch.nn import ModuleList, ReLU
import torchlight
from sklearn.metrics import confusion_matrix
import seaborn as sns

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()

        #self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[1][1][0].gcn.conv
        first_layer.register_backward_hook(hook_function)

    # def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model._modules.items():
            if isinstance(module, ModuleList):
                for each_module in module:
                    each_module.relu.register_backward_hook(relu_backward_hook_function)
                    each_module.relu.register_forward_hook(relu_forward_hook_function)

    #
    # def update_relus(self):
    #     """
    #         Updates relu activation functions so that
    #             1- stores output in forward pass
    #             2- imputes zero for gradient values that are less than zero
    #     """
    #     def relu_backward_hook_function(module, grad_in, grad_out):
    #         """
    #         If there is a negative gradient, change it to zero
    #         """
    #         # Get last forward output
    #         corresponding_forward_output = self.forward_relu_outputs[-1]
    #         corresponding_forward_output[corresponding_forward_output > 0] = 1
    #         modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
    #         del self.forward_relu_outputs[-1]  # Remove last forward output
    #         return (modified_grad_out,)
    #
    #     def relu_forward_hook_function(module, ten_in, ten_out):
    #         """
    #         Store results of forward pass
    #         """
    #         self.forward_relu_outputs.append(ten_out)
    #
    #     # Loop through layers, hook up ReLUs
    #     for pos, module in self.model.features._modules.items():
    #         if isinstance(module, ReLU):
    #             module.register_backward_hook(relu_backward_hook_function)
    #             module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):

        # Forward pass
        output, _ = self.model(input_image)

        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.cuda.FloatTensor(output.size()).zero_()
        for idx in range(output.shape[0]):
            one_hot_output[idx, target_class[idx]] = 1
        # Backward pass
        output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


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


def find_all_substr(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def get_best_epoch_and_accuracy(path_to_model_files):
    all_models = os.listdir(path_to_model_files)
    while '_' not in all_models[-1]:
        all_models = all_models[:-1]
    best_model = all_models[-1]
    all_us = list(find_all_substr(best_model, '_'))
    return int(best_model[5:all_us[0]]), float(best_model[all_us[0]+4:all_us[1]])


def plot_confusion_matrix(confusion_matrix, title='CM_96.9', fontsize=50):
    # mpl.style.use('seaborn')
    mpl.style.use('seaborn-v0_8')  # ✅ hoặc comment dòng này nếu không quan trọng style

    rcParams['text.usetex'] = False
    rcParams['axes.titlepad'] = 20

    columns = ('Angry', 'Neutral', 'Happy', 'Sad')
    rows = columns
    fig, ax = plt.subplots()

    # Set colors
    colors = np.empty((4, 4))
    colors[0] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['goldenrod'], 1.0))
    colors[1] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['bisque'], 1.0))
    colors[2] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['paleturquoise'], 1.0))
    colors[3] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['limegreen'], 1.0))
    # colors[4] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['lightpink'], 1.0))
    # colors[5] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['hotpink'], 1.0))
    # colors[6] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['mistyrose'], 1.0))
    # colors[7] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['lightsalmon'], 1.0))
    # colors[8] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['lavender'], 1.0))
    # colors[9] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['cornflowerblue'], 1.0))

    n_rows = len(confusion_matrix)
    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        # plt.bar(index, confusion_matrix[row], bar_width, bottom=y_offset,
        #                                                 color=colors[row])
        y_offset = y_offset + confusion_matrix[row]
        cell_text.append(['%d' % (x) for x in confusion_matrix[row]])

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom')
    the_table.set_fontsize(fontsize)
    the_table.scale(1, fontsize/7)

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2,
                        bottom=0.1,
                        top=0.99)

    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    plt.ylabel("\# predictions of each class", fontsize=fontsize)
    plt.xticks([])
    os.makedirs('figures', exist_ok=True)  # đảm bảo thư mục tồn tại
    fig.savefig('figures/'+title+'.png', bbox_inches='tight')


class Processor(object):
    """
        Processor for gait generation
    """

    def __init__(self, args, data_loader, C, num_classes, graph_dict, device='cuda:0', verbose=True):

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
        self.verbose = verbose
        self.io = torchlight.IO(
            self.args.work_dir,
            save_log=self.args.save_log,
            print_log=self.args.print_log)

        # model
        if not os.path.isdir(self.args.work_dir):
            os.mkdir(self.args.work_dir)
        self.model = classifier.Classifier(C, num_classes, graph_dict)
        self.model.cuda('cuda:0')
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        self.best_loss = math.inf
        self.step_epochs = [math.ceil(float(self.args.num_epoch * x)) for x in self.args.step]
        self.best_epoch = None
        self.best_accuracy = np.zeros((1, np.max(self.args.topk)))
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

        # if self.args.optimizer == 'SGD' and\
        if self.meta_info['epoch'] in self.step_epochs:
            lr = self.args.base_lr * (
                    0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.step_epochs)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr

    def show_epoch_info(self):

        for k, v in self.epoch_info.items():
            if self.verbose:
                self.io.print_log('\t{}: {}'.format(k, v))
        if self.args.pavi_log:
            if self.verbose:
                self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):

        if self.meta_info['iter'] % self.args.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)
            if self.verbose:
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
        if self.verbose:
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

        for data, label in loader:
            # get data
            data = data.float().to(self.device)
            label = label.long().to(self.device)

            # forward
            output, _ = self.model(data)
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
        if self.verbose:
            self.io.print_timer()
        # for k in self.args.topk:
        #     self.calculate_topk(k, show=False)
        # if self.accuracy_updated:
            # self.model.extract_feature()

    def per_test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:

            # get data
            data = data.float().to(self.device)
            label = label.long().to(self.device)

            # inference
            with torch.no_grad():
                output, _ = self.model(data)
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
            for k in self.args.topk:
                self.show_topk(k)

    def train(self):
        print('Training Start:')
        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            self.meta_info['epoch'] = epoch
            print('Epoch: {:d}'.format(epoch))
            # training
            if self.verbose:
                self.io.print_log('Training epoch: {}'.format(epoch))
            self.per_train()
            if self.verbose:
                self.io.print_log('Done.')

            # evaluation
            if (epoch % self.args.eval_interval == 0) or (
                    epoch + 1 == self.args.num_epoch):
                if self.verbose:
                    self.io.print_log('Eval epoch: {}'.format(epoch))
                self.per_test()
                if self.verbose:
                    self.io.print_log('Done.')

            # save model and weights
            if self.accuracy_updated:
                torch.save(self.model.state_dict(),
                           os.path.join(self.args.work_dir,
                                        'epoch{}_acc{:.2f}_model.pth.tar'.format(epoch, self.best_accuracy.item())))
                if self.epoch_info['mean_loss'] < self.best_loss:
                    self.best_loss = self.epoch_info['mean_loss']
                self.best_epoch = epoch

        self.plot_accuracy()
        self.plot_loss()
        self.plot_loss1()




    def test(self):

        # the path of weights must be appointed
        if self.args.weights is None:
            raise ValueError('Please appoint --weights.')
        if self.verbose:
            self.io.print_log('Model:   {}.'.format(self.args.model))
            self.io.print_log('Weights: {}.'.format(self.args.weights))

        # evaluation
        if self.verbose:
            self.io.print_log('Evaluation Start:')
        self.per_test()
        if self.verbose:
            self.io.print_log('Done.\n')

        # save the output of model
        if self.args.save_result:
            result_dict = dict(
                zip(self.data_loader['test'].dataset.sample_name,
                    self.result))
            self.io.save_pkl(result_dict, 'test_result.pkl')

    def smap(self):
        # self.model.eval()
        loader = self.data_loader['test']

        for data, label in loader:

            # get data
            data = data.float().to(self.device)
            label = label.long().to(self.device)

            GBP = GuidedBackprop(self.model)
            guided_grads = GBP.generate_gradients(data, label)

    def load_best_model(self):
        # if self.best_epoch is None:
        #     self.best_epoch, best_accuracy = get_best_epoch_and_accuracy(self.args.work_dir)
        # else:
        #     best_accuracy = self.best_accuracy.item()

        filename = os.path.join((r'D:\PBL4\STEP\classifier_stgcn_real_only\model_73\features_3D_my_82\epoch304_acc96.88_model.pth.tar'))
        self.model.load_state_dict(torch.load(filename))

    def generate_predictions(self, data, num_classes, joints, coords):
        # fin = h5py.File('../data/features'+ftype+'.h5', 'r')
        # fkeys = fin.keys()
        
        labels_pred = np.zeros(data.shape[0])
        output = np.zeros((data.shape[0], num_classes))
        for i, each_data in enumerate(zip(data)):
            # get data
            each_data = each_data[0]
            each_data = np.reshape(each_data, (1, each_data.shape[0], joints, coords, 1))
            each_data = np.moveaxis(each_data, [1, 2, 3], [2, 3, 1])
            each_data = torch.from_numpy(each_data).float().to(self.device)
            # get label
            with torch.no_grad():
                output_torch, _ = self.model(each_data)
                output[i] = output_torch.detach().cpu().numpy()
                labels_pred[i] = np.argmax(output[i])
        return labels_pred, output
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
        plot_confusion_matrix(confusion_matrix)

    def save_best_feature(self, ftype, data, joints, coords):
        if self.best_epoch is None:
            self.best_epoch, best_accuracy = get_best_epoch_and_accuracy(self.args.work_dir)
        else:
            best_accuracy = self.best_accuracy.item()
        filename = os.path.join(self.args.work_dir,
                                'epoch{}_acc{:.2f}_model.pth.tar'.format(self.best_epoch, best_accuracy))
        self.model.load_state_dict(torch.load(filename))
        features = np.empty((0, 64))
        fCombined = h5py.File('../data/features'+ftype+'.h5', 'r')
        fkeys = fCombined.keys()
        dfCombined = h5py.File('../data/deepFeatures'+ftype+'.h5', 'w')
        for i, (each_data, each_key) in enumerate(zip(data, fkeys)):

            # get data
            each_data = np.reshape(each_data, (1, each_data.shape[0], joints, coords, 1))
            each_data = np.moveaxis(each_data, [1, 2, 3], [2, 3, 1])
            each_data = torch.from_numpy(each_data).float().to(self.device)

            # get feature
            with torch.no_grad():
                _, feature = self.model(each_data)
                fname = [each_key][0]
                dfCombined.create_dataset(fname, data=feature)
                features = np.append(features, np.array(feature).reshape((1, feature.shape[0])), axis=0)
        dfCombined.close()
        return features

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
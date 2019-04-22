import numpy as np
from sklearn.datasets import fetch_openml
from network import ConvNet2
from network2 import ConvNet
import matplotlib.pyplot as plt
import pickle as pickle
import os
import cv2
from skimage.io import imread, imsave
from skimage.feature import blob_log
from skimage.color import rgb2gray
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from scipy.misc import imresize
import utils


def load_data(flatten=False):
    data = {}
    path1 = './train_numbers/'
    path = './test_numbers/'
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for img in os.listdir(path):
        filename = img.strip('.jpg')
        if filename != 'DS_Store':
            if len(filename) < 4:
                full_path = path + img
                image = imread(full_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                train_data.append(image)

                # train_feature = np.zeros(11)
                # train_feature[int(img[0]) - 2] = 1
                # train_labels.append(train_feature)

                train_labels.append(int(img[0]) - 2)
            else:
                full_path = path + img
                image = imread(full_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                train_data.append(image)

                # train_feature = np.zeros(11)
                # train_feature[int(img[0:2]) - 2] = 1
                # train_labels.append(train_feature)

                train_labels.append(int(img[0:2]) - 2)

    
    for img in os.listdir(path1):
        filename = img.strip('.jpg')
        if filename != 'DS_Store':
            if len(filename) < 4:
                full_path = path1 + img
                image = imread(full_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                test_data.append(image)

                # test_feature = np.zeros(11)
                # test_feature[int(img[0]) - 2] = 1
                # test_labels.append(test_feature)

                test_labels.append(int(img[0]) - 2)

            else:
                full_path = path1 + img
                image = imread(full_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                test_data.append(image)

                # test_feature = np.zeros(11)
                # test_feature[int(img[0:2]) - 2] = 1
                # test_labels.append(test_feature)

                test_labels.append(int(img[0:2]) - 2)
   
    test_data = np.array(test_data)
    train_data = np.array(train_data)

   
    data['X_train'] = train_data
    data['y_train'] = np.array(train_labels)
    data['X_test'] = test_data
    data['y_test'] = np.array(test_labels)
    data['X_val'] = train_data[:5]
    data['y_val'] = np.array(train_labels[:5])

    return data

def plot_performance(num_epochs, val_losses, train_losses, train_accuracies, val_accuracies):

    plt.xlabel('epoch number')
    plt.ylabel('avg loss')

    plt.plot(num_epochs, val_losses, color='blue', linestyle='dashed', label='val_loss')
    plt.plot(num_epochs, train_losses, color='yellow', linestyle='dashed', label='train_loss')
    plt.legend(loc='upper right')
    plt.savefig('visualization2.png')
    plt.show()

    plt.ylabel('average accuracy')
    plt.plot(num_epochs, train_accuracies, color='yellow', linestyle='dashed', label='training_accuracy')
    plt.plot(num_epochs, val_accuracies, color='blue', linestyle='dashed', label='val_accuracy')
    plt.legend(loc='upper right')
    plt.savefig('visualization3.png')
    plt.show()

def load_checkpoint():
    checkpoints = []
    for i in range(0, 3):
        filename = '%s_epoch_%d.pkl' % ('extras', i)
        with open(filename, 'rb') as cp_file:
            cp = pickle.load(cp_file)
            checkpoints.append(cp)
    
    return checkpoints

def predictions(logits):
    """
    Given the network output, determines the predicted class index

    Returns:
        the predicted class output as a PyTorch Tensor
    """
    tensors = []
    max_val = logits[0]
    max_val = max_val[0]
    best_index = 0
    for logit in logits:
        # max_val = logit[0]
        # best_index = 0
        # print(logit)
        for count, val in enumerate(logit):
            # print(val)
            # print(float(val))
            # val = float(val)
            if val > max_val:
                max_val = val
                best_index = count
            # tensors.append(best_index)

    
    # tensors = torch.Tensor(tensors)
    # print(tensors)
    # tensors = torch.Tensor(best_index)

    return best_index

    return tensors

def _train_epoch(X_train, y_train, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    model = model.train()
    for i, image in enumerate(X_train):
        label = y_train[i]
        label = torch.from_numpy(np.asarray(label))
        label = label.type(torch.LongTensor)
        label = torch.tensor([label])

        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        image = image.unsqueeze(0)

        optimizer.zero_grad()
        output = model(image)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

def _evaluate_epoch(X_train, y_train, X_val, y_val, model, criterion, epoch):
    """
    Evaluates the `model` on the train and validation set.
    """
    print('evaluating epoch')
    y_true, y_pred = [], []
    correct, total = 0, 0
    running_loss = []
    losses = 0
    for i, image in enumerate(X_train):
        with torch.no_grad():
            label = y_train[i]
            label = torch.from_numpy(np.asarray(label))
            label = label.type(torch.LongTensor)
            label = torch.tensor([label])

            image = torch.from_numpy(image)
            image = image.type(torch.FloatTensor)
            image = image.unsqueeze(0)

            output = model(image)
            # predicted = predictions(output.data)
            # y_true.append(label)
            # y_pred.append(predicted)
            total += label.size(0)

            label = label.float()
            # correct += (predicted == label).sum().item()
            # correct += predicted == label
            correct_output = np.argmax(output)
            if float(label) == float(correct_output):
                correct += 1
            # print('correct is : ' + str((predicted == label).sum().item()))
            # loss = criterion(output, label)
            # losses += loss.item()

            label = label.long()
            running_loss.append(criterion(output, label).item())
            

    train_loss = np.mean(running_loss)
    train_acc = correct / total
    # train_acc = losses / total
    print('training loss: ' + str(train_loss))
    print('train acc: ' + str(train_acc))
    y_true, y_pred = [], []
    correct, total = 0, 0
    running_loss = []
    for i, image in enumerate(X_val):
        with torch.no_grad():
            label = y_val[i]
            label = torch.from_numpy(np.asarray(label))
            label = label.type(torch.LongTensor)
            label = torch.tensor([label])

            image = torch.from_numpy(image)
            image = image.type(torch.FloatTensor)
            image = image.unsqueeze(0)

            output = model(image)
            predicted = predictions(output.data)
            y_true.append(label)
            y_pred.append(predicted)
            total += label.size(0)
            label = label.float()
            correct += (predicted == label).sum().item()
            label = label.long()
            running_loss.append(criterion(output, label).item())
    val_loss = np.mean(running_loss)
    val_acc = correct / total
    print('val loss: ' + str(val_loss))
    print('val acc: ' + str(val_acc))
    # stats.append([val_acc, val_loss, train_acc, train_loss])
    # utils.log_cnn_training(epoch, stats)
    # utils.update_cnn_training_plot(axes, epoch, stats)

def test(X_test, y_test, model, criterion):
    '''
    Function for testing.
    '''
    correct = 0.
    total = 0
    with torch.no_grad():
        model = model.eval()
        for i, image in enumerate(X_test):
            label = y_train[i]
            label = torch.from_numpy(np.asarray(label))
            label = label.type(torch.LongTensor)
            label = torch.tensor([label])

            image = torch.from_numpy(image)
            image = image.type(torch.FloatTensor)
            image = image.unsqueeze(0)

            output = model(image)
            total += label.size(0)

            label = label.float()
            correct_output = np.argmax(output)
            if float(label) == float(correct_output):
                correct += 1

            label = label.long()
    print('test accuracy is')
    print(correct / total)

def resize(X):

    resized = []
    for img in X:
        image_dim = 50
        resize = imresize(img, (image_dim, image_dim, 3), interp='bicubic')
        resized.append(resize)

    resized = np.array(resized)
    return resized

if __name__=="__main__":

    data = load_data()
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    X_val = data['X_val']
    y_val = data['y_val']

    X_train = resize(X_train)
    X_test = resize(X_test)
    X_val = resize(X_val)
    X_train = X_train.transpose(0, 3, 1, 2)
    X_test = X_test.transpose(0, 3, 1, 2)
    X_val = X_val.transpose(0, 3, 1, 2)

    model = ConvNet()
    print('finished model')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(0,20):

        _train_epoch(X_train, y_train, model, criterion, optimizer)

        _evaluate_epoch(X_train, y_train, X_val, y_val, model, criterion, epoch)

        test(X_test, y_test, model, criterion)

        print('finished epoch ' + str(epoch))
    
    print('finished training')



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

                train_feature = np.zeros(11)
                train_feature[int(img[0]) - 2] = 1
                train_labels.append(train_feature)

                # train_labels.append(int(img[0]))
                # print(int(img[0]))
            else:
                full_path = path + img
                # print(full_path)
                image = imread(full_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                train_data.append(image)

                train_feature = np.zeros(11)
                train_feature[int(img[0:2]) - 2] = 1
                train_labels.append(train_feature)

                # train_labels.append(int(img[0:2]))
                # print(int(img[0:2]))

    
    for img in os.listdir(path1):
        filename = img.strip('.jpg')
        if filename != 'DS_Store':
            if len(filename) < 4:
                full_path = path1 + img
                # print(full_path)
                image = imread(full_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                test_data.append(image)

                test_feature = np.zeros(11)
                test_feature[int(img[0]) - 2] = 1
                test_labels.append(test_feature)

                # test_labels.append(int(img[0]))
                # print(int(img[0]))
            else:
                full_path = path1 + img
                # print(full_path)
                image = imread(full_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                test_data.append(image)

                test_feature = np.zeros(11)
                test_feature[int(img[0:2]) - 2] = 1
                test_labels.append(test_feature)

                # test_labels.append(int(img[0:2]))
                # print(int(img[0:2]))
   
    test_data = np.array(test_data)
    train_data = np.array(train_data)
    # y = y.reshape(y.shape[0],1)
    # if not flatten:
    #     test_data = test_data.reshape(test_data.shape[0], 28, 28)
    #     test_data = test_data[:, np.newaxis, :, :]
    #     train_data = train_data.reshape(train_data.shape[0], 28, 28)
    #     train_data = train_data[:, np.newaxis, :, :]

   
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

def _train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    for i, (X, y) in enumerate(data_loader):
        optimizer.zero_grad()

        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

def resize(X):

    resized = []
    for img in X:
        image_dim = 28
        resize = imresize(img, (image_dim, image_dim, 3), interp='bicubic')
        resized.append(resize)

    resized = np.array(resized)
    return resized
               

# def train():
#     # load data
#     print('loading data')

    
#     data = load_data()
#     model = ConvNet()
#     criterion = nn.LogSoftmax()

   


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

    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
    for epoch in range(0,5):
        i = 0
        for image,label in X_train, y_train:
            out = model(image)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if (i+1)%100 == 0:
                print("Epoch[{}/{}], step[{}/{}], loss={:0.4f}".format(epoch+1,5,i+1,len(X_train),loss.item()))
            i += 1

    torch.save(model.state_dict(), 'model.ckpt')

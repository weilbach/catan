import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.misc import imread, imresize
import cv2
from skimage.io import imsave
import pickle as pickle


def load_data(flatten=False):
    data = {}
    path = './train_numbers/'
    path1 = './test_numbers/'
    path2 = './val_numbers/'
    path3 = './perfect_numbers/'
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    val_data = []
    val_labels = []
    perf_data = []
    perf_labels = []
    for img in os.listdir(path):
        filename = img.strip('.jpg')
        filenames = img.split('_')
        if filename != 'DS_Store':
            number = filenames[0]
            full_path = path + img
            image = imread(full_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            train_data.append(image)
            train_labels.append(int(number) - 2)
    
    for img in os.listdir(path1):
        filename = img.strip('.jpg')
        filenames = img.split('_')
        if filename != 'DS_Store':
            number = filenames[0]
            full_path = path1 + img
            image = imread(full_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            test_data.append(image)
            test_labels.append(int(number) - 2)
    
    for img in os.listdir(path2):
        filename = img.strip('.jpg')
        filenames = img.split('_')
        if filename != 'DS_Store':
            number = filenames[0]
            full_path = path2 + img
            image = imread(full_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            val_data.append(image)
            val_labels.append(int(number) - 2)
    
    for img in os.listdir(path3):
        filename = img.strip('.jpg')
        filenames = img.split('_')
        if filename != 'DS_Store':
            number = filenames[0]
            full_path = path3 + img
            image = imread(full_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            perf_data.append(image)
            perf_labels.append(int(number) - 2)

   
    test_data = np.array(test_data)
    train_data = np.array(train_data)
    val_data = np.array(val_data)
    perf_data = np.array(perf_data)

   #put all the data in a dictionary
    data['X_train'] = train_data
    data['y_train'] = train_labels
    data['X_test'] = test_data
    data['y_test'] = np.array(test_labels)
    data['X_val'] = val_data
    data['y_val'] = np.array(val_labels)
    data['X_perf'] = perf_data
    data['y_perf'] = np.array(perf_labels)

    return data

def plot_performance(num_epochs, val_losses, train_losses, val_accuracies, train_accuracies):

    plt.xlabel('epoch number')
    plt.ylabel('avg loss')

    plt.plot(num_epochs, val_losses, color='blue', linestyle='dashed', label='val_loss')
    plt.plot(num_epochs, train_losses, color='yellow', linestyle='dashed', label='train_loss')
    plt.legend(loc='upper right')
    plt.savefig('visualization4.png')
    plt.show()

    plt.ylabel('accuracy')
    plt.plot(num_epochs, train_accuracies, color='yellow', linestyle='dashed', label='training_accuracy')
    plt.plot(num_epochs, val_accuracies, color='blue', linestyle='dashed', label='val_accuracy')
    plt.legend(loc='upper right')
    plt.savefig('visualization5.png')
    plt.show()

def load_checkpoint(name, epoch):

    checkpoints = []
    filename = '%s_epoch_%d.pkl' % (name, epoch)
    with open(filename, 'rb') as cp_file:
        cp = pickle.load(cp_file)
        checkpoints.append(cp)
    
    return checkpoints

def save_checkpoint(model, decay, epoch, val_acc_history, train_acc_history, val_loss_history, train_loss_history):
    checkpoint = {
        'model': model,
        'lr_decay': decay,
        'epoch': epoch,
        'val_acc_history': val_acc_history,
        'train_acc_history': train_acc_history,
        'val_loss_history': val_loss_history,
        'train_loss_history': train_loss_history
    }
    filename = '%s_epoch_%d.pkl' % ('bunk', epoch)
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)

def resize(X):

    resized = []
    for img in X:
        image_dim = 50
        resize = imresize(img, (image_dim, image_dim, 3), interp='bicubic')
        resized.append(resize)

    resized = np.array(resized)
    return resized


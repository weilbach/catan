import numpy as np
from sklearn.datasets import fetch_openml

from network import ConvNet2
from network2 import ConvNet
from solver import Solver
import matplotlib.pyplot as plt
import pickle as pickle
import os
import cv2
from skimage.io import imread, imsave
from skimage.feature import blob_log
from skimage.color import rgb2gray


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
                # print(full_path)
                image = imread(full_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                train_data.append(image)

                # train_feature = np.zeros(11)
                # train_feature[int(img[0]) - 2] = 1
                # train_labels.append(train_feature)

                train_labels.append(int(img[0]))
                print(int(img[0]))
            else:
                full_path = path + img
                # print(full_path)
                image = imread(full_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                train_data.append(image)

                # train_feature = np.zeros(11)
                # train_feature[int(img[0:2]) - 2] = 1
                # train_labels.append(train_feature)

                train_labels.append(int(img[0:2]))
                print(int(img[0:2]))

    
    for img in os.listdir(path1):
        filename = img.strip('.jpg')
        if filename != 'DS_Store':
            if len(filename) < 4:
                full_path = path1 + img
                # print(full_path)
                image = imread(full_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                test_data.append(image)

                # test_feature = np.zeros(11)
                # test_feature[int(img[0]) - 2] = 1
                # test_labels.append(test_feature)

                test_labels.append(int(img[0]))
                # print(int(img[0]))
            else:
                full_path = path1 + img
                # print(full_path)
                image = imread(full_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                test_data.append(image)

                # test_feature = np.zeros(11)
                # test_feature[int(img[0:2]) - 2] = 1
                # test_labels.append(test_feature)

                test_labels.append(int(img[0:2]))
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

def get_graphs(data):
    checkpoints = load_checkpoint()
    train_losses = []
    val_losses = []
    for cp in checkpoints:
        model = cp['model']
        solver = Solver(model, data, update_rule='sgd',
                    optim_config={'learning_rate': 1e-2,},
                    lr_decay=1.0, num_epochs=10,
                    batch_size=16, print_every=1)
        _ = solver.check_accuracy(data['X_train'], data['y_train'], num_samples=1000)
        #lets print the val loss history and the self.loss history for all of them when it's done
        for i in range(0, len(solver.loss_history), 3125):
          # val_loss = np.mean(val_losses[i:i+iterations_per_epoch])
          train_loss = np.mean(solver.loss_history[i:i+3125])
          # avg_val_loss.append(val_loss)
          train_losses.append(train_loss)

        #this might be all i need gonna need to print to make sure though
        # train_losses.append(np.mean(solver.loss_history))
        val_losses = solver.val_loss
    
    return train_losses, val_losses
               

def train():
    # load data
    print('loading data')

    
    data = load_data()
    model = ConvNet2()

    # intialize solver
    print('initializing solver')

    solver = Solver(model, data, update_rule='sgd',
                    optim_config={'learning_rate': 1e-3,},
                    lr_decay=1e-2, num_epochs=3,
                    batch_size=1, print_every=1)


    # # start training
    print('starting training')
    solver.train()

    # # plot losses and accuracies
    # plot_performance(epochs, validation_losses, training_losses, train_acc_hist, val_acc_hist)

    # report test accuracy
    print(data['y_test'])
    acc = solver.check_accuracy(data['X_test'], data['y_test'])
    print("Test accuracy: {}".format(acc))

    


if __name__=="__main__":
    train()

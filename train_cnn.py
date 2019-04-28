import numpy as np
from network import ConvNet2
from network2 import ConvNet
import matplotlib.pyplot as plt
from dataset import plot_performance, load_checkpoint, resize, load_data, save_checkpoint
import os
import cv2
from skimage.io import imread, imsave
from skimage.color import rgb2gray
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import utils
import collections



def _train_epoch(X_train, y_train, model, criterion, optimizer, batch_size, iterations):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    model = model.train()

    #create batch size for sgd
    num_train = X_train.shape[0]
    batch_mask = np.random.choice(num_train, batch_size)
    train_batch = X_train[batch_mask]
    #lmao why is that not already an np array
    y_train = np.array(y_train)
    batch_labels = np.asarray(y_train[batch_mask])

    for it in range(0, iterations):
        for i, image in enumerate(train_batch):
            label = batch_labels[i]
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
    correct, total = 0, 0
    running_loss = []
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
            total += label.size(0)

            label = label.float()
            correct_output = np.argmax(output)
            if float(label) == float(correct_output):
                correct += 1

            label = label.long()
            running_loss.append(criterion(output, label).item())

    train_loss = np.mean(running_loss)
    train_acc = correct / total
    print('training loss: ' + str(train_loss))
    print('train acc: ' + str(train_acc))
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
            total += label.size(0)

            label = label.float()
            correct_output = np.argmax(output)
            if float(label) == float(correct_output):
                correct += 1

            label = label.long()
            running_loss.append(criterion(output, label).item())

    val_loss = np.mean(running_loss)
    val_acc = correct / total
    print('val loss: ' + str(val_loss))
    print('val acc: ' + str(val_acc))
    
    return val_loss, train_loss, val_acc, train_acc

def test(X_test, y_test, model):
    '''
    Function for testing.
    '''
    correct = 0.
    total = 0
    outputs = []
    labels = []
    predicted = []
    with torch.no_grad():
        model = model.eval()
        for i, image in enumerate(X_test):
            label = y_test[i]
            label = torch.from_numpy(np.asarray(label))
            label = label.type(torch.LongTensor)
            label = torch.tensor([label])

            image = torch.from_numpy(image)
            image = image.type(torch.FloatTensor)
            image = image.unsqueeze(0)

            output = model(image)
            total += label.size(0)

            label = label.float()
            pred = np.argmax(output)
            if float(label) == float(pred):
                correct += 1

            outputs.append(output)
            labels.append(label)
            predicted.append(float(pred))
    print('test accuracy is')
    print(correct / total)

    return outputs, labels, predicted

def final_prediction(outputs, labels, predicted):

    #this is quite the work in progress
    
    all_numbers = []
    found_indices = []
    ones = {}
    twos = {}
    

    count = 0
    #should go through outputs and select most positive ones to numbers
    while(count != 18):
        highest_guess = 0
        best_index = None
        for ind, output in enumerate(outputs):
            ind_best = np.argmax(output[0])
            best = output[0][ind_best]
            if best > highest_guess:
                highest_guess = best
                best_index = ind
        
        if best_index == 0 or best_index == 10:
            if len(ones) == 0:
                ones[best_index] = ind
            else:
                output[0][best_index] = -100.0
        else:
            if len(twos[best_index]) < 2:
                if best_index not in twos:
                    twos[best_index] == [ind]
                else:
                    twos[best_index].append(ind)
            else:
                output[0][best_index] = -100
        
        count2 = 0
        for key in twos:
            for _ in twos[key]:
                count2 += 1
        
        count = len(ones) + count2
    
    return outputs

def test_final(outputs, labels):
    correct = 0.
    total = 0
    # outputs = []
    # labels = []
    # predicted = []
    with torch.no_grad():
        model = model.eval()
        for i, output in enumerate(outputs):
            label = y_test[i]
            label = torch.from_numpy(np.asarray(label))
            label = label.type(torch.LongTensor)
            label = torch.tensor([label])

            # image = torch.from_numpy(image)
            # image = image.type(torch.FloatTensor)
            # image = image.unsqueeze(0)

            # output = model(image)
            total += label.size(0)

            label = label.float()
            pred = np.argmax(output)
            if float(label) == float(pred):
                correct += 1

            # outputs.append(output)
            # labels.append(label)
            # predicted.append(float(pred))
    print('test accuracy is')
    print(correct / total)
        




if __name__=="__main__":

    data = load_data()
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    X_val = data['X_val']
    y_val = data['y_val']
    X_perf = data['X_perf']
    y_perf = data['y_perf']

    X_train = resize(X_train)
    X_test = resize(X_test)
    X_val = resize(X_val)
    X_perf = resize(X_perf)
    X_train = X_train.transpose(0, 3, 1, 2)
    X_test = X_test.transpose(0, 3, 1, 2)
    X_val = X_val.transpose(0, 3, 1, 2)
    X_perf = X_perf.transpose(0, 3, 1, 2)

    print(len(X_train))
    print(len(X_test))
    print(len(X_val))
    print(len(X_perf))

    model = ConvNet()
    print('finished model')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    val_loss_history = []
    train_loss_history = []
    val_acc_history = []
    train_acc_history = []
    num_epochs = 75

    # for epoch in range(0, num_epochs):
    #     batch_size = 16

    #     _train_epoch(X_train, y_train, model, criterion, optimizer, batch_size, 10)

    #     val_loss, train_loss, val_acc, train_acc = _evaluate_epoch(X_train, y_train, X_val, y_val, model, criterion, epoch)
    #     val_loss_history.append(val_loss)
    #     train_loss_history.append(train_loss)
    #     val_acc_history.append(val_acc)
    #     train_acc_history.append(train_acc)

    #     print('finished epoch ' + str(epoch))

    # save_checkpoint(model, .001, epoch, val_acc_history, train_acc_history, val_loss_history, train_loss_history)
    
    checkpoints = load_checkpoint('yunk', 74)
    model = checkpoints[0]['model']
    print('finished training')
    test(X_test, y_test, model) #test accuracy


    
    print('testing perfect data')
    # checkpoints = load_checkpoint('bunk', epoch)
    # model = checkpoints[0]['model']

    outputs, labels, predicted = test(X_perf, y_perf, model)

    #collect final predictions, outputs will change
    outputs = final_prediction(outputs, labels, predicted)
    test_final(outputs, labels)

    #graphing
    # epoch_list = list(range(0, 60))

    # plot_performance(epoch_list, val_loss_history, train_loss_history, val_acc_history, train_acc_history)




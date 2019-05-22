import numpy as np
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
    #this function is inspired by an EECS 445 project done by Justin Weilbach

    #I found online that people do this but I don't think it did anything
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

    #this function is inspired by an EECS 445 project done by Justin Weilbach

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

    #this function is slightly inspired by an EECS 445 project done by Justin Weilbach

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
    #this function is an algorithm for reclassifying predictions

    ones = {0: [], 10: []}
    twos = {1: [], 2: [],3: [], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    nums = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
    
    for ind, tensor in enumerate(outputs):
        highest_guess = 0
        best_index = None
        for index, values in enumerate(tensor[0]):
            nums[index].append(values)
    
    count = 0
    used_indices = []
    #iterate until all 18 tensor predictions have been used
    while(count != 18):
        highest_guess = -100.0
        best_index = None
        num_in_use = None
        for key in nums:
            for ind, val in enumerate(nums[key]):
                #ensure one tensor is not used more than once
                if ind not in used_indices:
                    if val > highest_guess:
                        highest_guess = val
                        best_index = ind
                        num_in_use = key
        
        if num_in_use == 0 or num_in_use == 10:
            if len(ones[num_in_use]) < 1:
                ones[num_in_use].append(best_index)
                used_indices.append(best_index)
            else:
                #this is lower than any prediction ever will be
                nums[num_in_use][best_index] = -100.0
        else:
            if len(twos[num_in_use]) < 2:
                twos[num_in_use].append(best_index)
                used_indices.append(best_index)
            else:
                nums[num_in_use][best_index] = -100.0
        
        count2 = 0
        #couldn't really find a better way to do this
        for key in twos:
            for _ in twos[key]:
                count2 += 1
        
        count = len(ones) + count2
    
    #check accuracy of refactored predictions
    correct = 0
    total = 0
    for ind, label in enumerate(labels):
        label = int(label)
        if label == 0 or label == 10:
            if ind in ones[label] :
                correct += 1
            total += 1
        else:
            if ind in twos[label]:
                correct += 1
            total += 1
    print(float(correct) / float(total))
    
        

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

    #this is data preprocessing
    #the transpose dimensions was inspired by Professor Fouhey
    X_train = resize(X_train)
    X_test = resize(X_test)
    X_val = resize(X_val)
    X_perf = resize(X_perf)
    X_train = X_train.transpose(0, 3, 1, 2)
    X_test = X_test.transpose(0, 3, 1, 2)
    X_val = X_val.transpose(0, 3, 1, 2)
    X_perf = X_perf.transpose(0, 3, 1, 2)

    #this ensures that the data has been processed correctly
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

    #comment this code out if you wish to simply load an old model
    for epoch in range(0, num_epochs):
        batch_size = 16

        _train_epoch(X_train, y_train, model, criterion, optimizer, batch_size, 10)

        val_loss, train_loss, val_acc, train_acc = _evaluate_epoch(X_train, y_train, X_val, y_val, model, criterion, epoch)
        val_loss_history.append(val_loss)
        train_loss_history.append(train_loss)
        val_acc_history.append(val_acc)
        train_acc_history.append(train_acc)

        print('finished epoch ' + str(epoch))

    save_checkpoint(model, .001, epoch, val_acc_history, train_acc_history, val_loss_history, train_loss_history)
    print('finished training')

    #comment this code out if you do not wish to load an old model
    checkpoints = load_checkpoint('bunk', 74)
    model = checkpoints[0]['model']
    val_acc_history = checkpoints[0]['val_acc_history']
    train_acc_history = checkpoints[0]['train_acc_history']
    val_loss_history = checkpoints[0]['val_loss_history']
    train_loss_history = checkpoints[0]['train_loss_history']
    
    
    #check test accuracy
    outputs, labels, predicted = test(X_test, y_test, model)
    
    print('testing perfect data')

    #this is for testing perfect data
    outputs, labels, predicted = test(X_perf, y_perf, model)

    #collect final predictions, outputs will change
    final_prediction(outputs, labels, predicted)

    #graphing
    epoch_list = list(range(0, num_epochs))
    plot_performance(epoch_list, val_loss_history, train_loss_history, val_acc_history, train_acc_history)




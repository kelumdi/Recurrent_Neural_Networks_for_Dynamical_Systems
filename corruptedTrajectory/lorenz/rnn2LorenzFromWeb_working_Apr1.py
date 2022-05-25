
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.utils.data
import time
import math
import numpy as np
import pandas as pd
import random
from scipy.integrate import odeint

np.random.seed(1)

# referred from : https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
class RNN2(nn.Module)  :
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN2, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining RNN layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, nonlinearity = 'relu')
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):

        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

# define true lorenz system
def lorenz_system(current_state, t):
    # define the system parameters sigma, rho, and beta
    sigma = 10.
    rho   = 28.
    beta  = 8./3.
    x, y, z = current_state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y  # delete y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# define defected lorenz system
def lorenz_systemCorrupted(current_state, t):
    # define the system parameters sigma, rho, and beta
    sigma = 10.
    rho   = 28.
    beta  = 8./3.
    x, y, z = current_state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z)  # deleted y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def getLorenz(k,stepSize ,timeStep ,time_points ,dataTotal):

    y1List = np.array([], dtype=np.float64).reshape(0, 3)
    y2List = np.array([], dtype=np.float64).reshape(0, 3)
    y2L = np.zeros((timeStep, 3))

    for j in range(0, dataTotal)  :

        # select random numbers between [-15,15] as initial condition for x y z
        initial_state = 30 * (np.random.uniform(low=1.0e-5, high=1.0, size=3) - 0.5)

        # get the points to plot, one chunk of time steps at a time, by integrating the system of equations
        y1L = odeint(lorenz_system, initial_state, time_points)
        y1List = np.concatenate((y1List, y1L), axis=0)

        lengthCorr = np.linspace(stepSize,k*stepSize,num=k)
        for i, e in enumerate(y1L):
            xx = odeint(lorenz_systemCorrupted, e, lengthCorr)
            y2L[i, :] = xx[k-1,:]
        y2List = np.concatenate((y2List, y2L), axis=0)

    y1L5 = [torch.tensor(y1List[x:x + timeStep]) for x in range(0, len(y1List), timeStep)]
    y2L5 = [torch.tensor(y2List[x:x + timeStep]) for x in range(0, len(y2List), timeStep)]

    #make a pair
    twoList = [(e2, e1) for i, e1 in enumerate(y1L5) for j, e2 in enumerate(y2L5) if i == j]

    return (y1L5, y2L5, twoList)

# get data
def getData(stepSize ,dataTotal ,k ,time_points, timeStep):
    (y1, y2, XY) = getLorenz(k,stepSize ,timeStep ,time_points ,dataTotal)
    train_xy = XY[0:int(len(XY) * (0.8))]  # select 80% of total data as train data
    test_xy = XY[int(len(XY) * (0.8)):len(XY)]  # select 20% of total data as test data
    return (train_xy, test_xy)


def trainRNN2MSE(tol ,k , model, n_epochs, lrRnn, train_loader, test_loader, batchSizeTrain ,batchSizeTest,seq_length, inputSize):

    criterion = nn.MSELoss() #mean squared loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lrRnn) #adam optimization

    # Training Run
    epochList = []
    lossListTrain = []
    lossListTest = []
    for epoch in range(1, n_epochs + 1):
        start = time.time()
        epochList.append(epoch)
        lossSumTrain = 0
        for batch_idx, (input, label) in enumerate(train_loader):
            # change shape : [b,inputSize,seqLength]-->[b,seqLength,inputSize]
            input = input.view(batchSizeTrain, seq_length, inputSize).float()
            label = label.view(batchSizeTrain, seq_length, inputSize).float(  )
            output, h_state = model(input)
            output = output.float()
            lossTrain = criterion(output ,label) #define loss
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            lossTrain.backward()  # Does backpropagation and calculates gradients
            optimizer.step()
            lossSumTrain = lossSumTrain + lossTrain.detach().numpy()
        lossTrainAvgForOneEpoch1 = lossSumTrain / len(train_loader.dataset)
        lossListTrain.append(lossTrainAvgForOneEpoch1)

        with torch.no_grad():
            lossSumTest = 0
            for batch_idx2, (inputTest, labelTest) in enumerate(test_loader):
                inputTest = inputTest.view(batchSizeTest, seq_length, inputSize).float()
                labelTest = labelTest.view(batchSizeTest, seq_length, inputSize).float()
                # Forward pass only to get logits/output
                outputTest, h_stateTest = model(inputTest)
                lossTest = criterion(outputTest, labelTest)
                lossSumTest = lossSumTest + lossTest.detach().numpy()
            lossTestAvgForOneEpoch1 = lossSumTest / len(test_loader.dataset  )#
            lossListTest.append(lossTestAvgForOneEpoch1)

        end = time.time()

        if (lossSumTrain / len(train_loader.dataset)) < tol: #stopping criteria
            break

        #print output every epoch
        if epoch % 1 == 0:
            print("Epoch :", epoch, '// One epoch time:', end - start, '// Train Error :', lossTrainAvgForOneEpoch1,
                  '// Test Error :', lossTestAvgForOneEpoch1)

    for i in range(0 ,batchSizeTest):

        # change shape : [b,seqLength,inputSize]-->[b,inputSize,seqLength]
        input = input.view(batchSizeTrain, inputSize, seq_length)
        label = label.view(batchSizeTrain, inputSize, seq_length)
        output = output.view(batchSizeTrain, inputSize, seq_length)
        inputTest = inputTest.view(batchSizeTest, inputSize, seq_length)
        labelTest = labelTest.view(batchSizeTest, inputSize, seq_length)
        outputTest = outputTest.view(batchSizeTest, inputSize, seq_length)

        # save outputs
        np.savetxt('./out/' + str(k) + '/Corrupted_train_{}.csv'.format(i), input[i].detach().numpy(),
                   delimiter=',')

        np.savetxt('./out/' + str(k) + '/True_train_{}.csv'.format(i), label[i].detach().numpy(), delimiter=',')

        np.savetxt('./out/' + str(k) + '/Recovered_train_{}.csv'.format(i), output[i].detach().numpy(),
                   delimiter=',')

        np.savetxt('./out/' + str(k) + '/Corrupted_test_{}.csv'.format(i), inputTest[i].detach().numpy(),
                   delimiter=',')

        np.savetxt('./out/' + str(k) + '/True_test_{}.csv'.format(i), labelTest[i].detach().numpy(), delimiter=',')

        np.savetxt('./out/' + str(k) + '/Recovered_test_{}.csv'.format(i), outputTest[i].detach().numpy(),
                   delimiter=',')

def main():

    dataTotal = 500 #total data
    epoch = 2  # number of epoch --> try 1000
    time_step = 5000  # time step

    k = 1  # corruption level, can select any number less than dataTotal
    tol = 1.0e-4 #tolerance for stopping criteria

    # define the time points to solve for, evenly spaced between the start and end times
    start_time = 0
    end_time = 25

    # define parameter values
    seqLength = 3
    input_size = time_step
    hidden_size = 256
    nLayer = 2
    output_size = input_size
    time_points = np.linspace(start_time, end_time, time_step)#0-50

    print('t :', time_points)
    stepSize = time_points[1] - time_points[0]
    print('stepSize :', stepSize)

    # get train and test data
    (train_xy, test_xy) = getData(stepSize, dataTotal, k, time_points, time_step)
    lrRnnAdam = 0.01 #learning rate for adam optimization

    # set batch size
    batchSizeTrain = int(dataTotal * 0.8)
    batchSizeTest = int(dataTotal * 0.2)

    train_loader = torch.utils.data.DataLoader(dataset=train_xy,
                                               batch_size=batchSizeTrain,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_xy,
                                              batch_size=batchSizeTest,
                                              shuffle=True)
    # set model
    model = RNN2(input_size=input_size, output_size=output_size, hidden_dim=hidden_size, n_layers=nLayer)

    #start training and testing
    trainRNN2MSE(tol, k, model, epoch,lrRnnAdam,train_loader, test_loader,batchSizeTrain,batchSizeTest,seqLength, input_size)


if __name__ == "__main__":
    # execute only if run as a script
    main()
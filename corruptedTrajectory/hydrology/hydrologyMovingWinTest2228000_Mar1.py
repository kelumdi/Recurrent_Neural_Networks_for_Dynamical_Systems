

from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data
import math
import numpy as np
import scipy.io
import time
import pickle

np.random.seed(0)

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#get data with window size shifted by one day
def getMovingWin(input2,lengthTimeBeforeMovingWin,shift,windowSize,iLimit):
    testInputL = []
    inputLTest = input2[0:lengthTimeBeforeMovingWin, ]
    for i in range(iLimit):
        bb = inputLTest[shift*i:shift*i+windowSize, ]
        testInputL.append(bb)
    return(testInputL)

# get trainInput, trainLabel, testInput, testLabel for train period and test period
def miniBatchDayShiftMonthMovingWinTrainMovingWinTest(input, label, numDay):

    windowSize = numDay
    shift = 1 #shifting by one day
    # 1.get trainInput & trainLabel for train period 0-2520 days
    # 1-1.get trainInput for train period 0-2520 days
    input2 = input.copy()
    inputTrain = input2[0:2520, ]
    lengthTimeBeforeMovingWin = inputTrain.shape[0]
    iLimit = int((lengthTimeBeforeMovingWin - windowSize) / shift) + 1
    trainInputL = getMovingWin(inputTrain, lengthTimeBeforeMovingWin, shift, windowSize, iLimit)

    # 1-2.get trainLabel for train period 0-2520 days
    label2 = label.copy()
    labelTrain = label2[0:2520, ]
    lengthTimeBeforeMovingWin = labelTrain.shape[0]
    iLimit = int((lengthTimeBeforeMovingWin - windowSize) / shift) + 1
    trainLabelL = getMovingWin(labelTrain, lengthTimeBeforeMovingWin, shift, windowSize, iLimit)

    # 2.get testInput & testLabel for train period 2520-3600 days
    # 2-1.get testInput for train period 2520-3600 days
    input2 = input.copy()
    inputTest = input2[0:3600, ]
    lengthTimeBeforeMovingWin = inputTest.shape[0]
    iLimit = int((lengthTimeBeforeMovingWin - windowSize) / shift) + 1
    testInputL = getMovingWin(inputTest, lengthTimeBeforeMovingWin, shift, windowSize, iLimit)

    # 2-2.get testLabel for train period 2520-3600 days
    label2 = label.copy()
    labelTest = label2[0:3600, ]
    lengthTimeBeforeMovingWin = labelTest.shape[0]
    iLimit = int((lengthTimeBeforeMovingWin - windowSize) / shift) + 1
    testLabelL = getMovingWin(labelTest, lengthTimeBeforeMovingWin, shift, windowSize, iLimit)
    return (trainInputL, trainLabelL, testInputL, testLabelL)

# List of site number and row number in the mofex table
def getRowNumTol(siteNum):
    if siteNum == 2273000:
        rowNum=86
        tol= 0
    elif siteNum == 2202500:
        rowNum= 80
        tol= 0
    elif siteNum == 2228000:
        rowNum= 84
        tol= 0
    elif siteNum == 2296750:
        rowNum= 87
        tol = 0
    elif siteNum == 2329000:#
        rowNum = 88
        tol= 0
    elif siteNum == 2349500:
        rowNum = 91
        tol = 0
    elif siteNum == 2365500:
        rowNum = 92
        tol = 0
    elif siteNum == 2375500:
        rowNum = 93
        tol= 0
    elif siteNum == 2475000:
        rowNum = 102
        tol = 0
    elif siteNum == 2479300:
        rowNum = 105
        tol = 0
    elif siteNum == 2492000:
        rowNum = 108
        tol = 0
    elif siteNum == 7375500:
        rowNum= 354
        tol= 0
    elif siteNum == 7378000:
        rowNum= 355
        tol= 0
    elif siteNum == 7378500:
        rowNum= 356
        tol = 0
    elif siteNum == 8189500:
        rowNum= 372
        tol=1.8e-4
    elif siteNum == 11224500:
        rowNum= 397
        tol= 0
    elif siteNum == 11138500:
        rowNum= 392
        tol = 0
    elif siteNum == 11025500:
        rowNum= 390
        tol= 0
    return(rowNum,tol)


# referred from : https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
class RNN2(nn.Module)  :
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN2, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining RNN layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True ,nonlinearity='relu')

        # Fully connected layer,
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)  # output=[seq_len, batch_size, num_features]

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

def getData2(data, label):
    y1List = data
    y2List = label
    y1L5 = [torch.Tensor(e) for i ,e in enumerate(y1List)]
    y2L5 = [torch.Tensor(e) for i, e in enumerate(y2List)]
    twoList = [(e1 ,e2) for i, e1 in enumerate(y1L5) for j, e2 in enumerate(y2L5) if i == j]
    return (y1L5, y2L5, twoList)

def getData(data, label):
    (y1, y2, XY) = getData2(data, label)
    return XY

# From given mopex438 data, get an information (QOBs,PET,PRECIP) of a specific site number
def getInputLabel(data ,rowNum):
    data92 = data[rowNum - 1, 0]

    d1 = data92[0]  # site Num
    d2 = data92[1]  # Year
    d3 = data92[2]  # month
    d4 = data92[3]  # day
    d5 = data92[4]  # QOBS
    d6 = data92[5]  # PRECIP(ation)
    d7 = data92[6]  # PET : Potential EvapoTranspiration
    d8 = data92[7]  # Temperature max
    d9 = data92[8]  # Temperature min

    for i, e in enumerate(d6.T):
        maxE = max(e)
        minE = min(e)
        d6L = []
        for i2, e2 in enumerate(e):
            nor = (e2 - minE) / (maxE -minE)
            d6L.append(nor)
    d6Num = np.array(d6L)
    d6Num = d6Num.reshape((d6Num.shape[0], 1))

    for i, e in enumerate(d7.T):
        maxE = max(e)
        minE = min(e)
        d7L = []
        for i2, e2 in enumerate(e):
            nor = (e2 - minE) / (maxE -minE)
            d7L.append(nor)
    d7Num = np.array(d7L)
    d7Num = d7Num.reshape((d7Num.shape[0], 1))

    input = np.concatenate([d6Num, d7Num], axis=1)
    label = d5
    return(input ,label ,d3)

# Moving Average Method : calculate average of the multiple predictions for each time step
def calculateOverlap_array(output3, lengthOriginal, sizeWindow, numTraj):
    overlapK = output3.copy()
    AA = []
    for i in range(0, lengthOriginal):

        if i < sizeWindow: #if index is less than window size
            l1 = np.linspace(0, i, i + 1)  # when i=2
            l2 = reversed(l1)
            AA1 = []
            for p, q in zip(l1, l2):
                overlapped = overlapK[int(p), :, int(q)]  # [8,]
                AA1.append(overlapped)
            AA.append(AA1)

        elif (i >= sizeWindow) and (i < numTraj - 1): #if window size < index < number of trajectory-1
            # l1=np.linspace(i-lengthMoving-1,i,lengthMoving)
            l1 = np.linspace(i - sizeWindow + 1, i, sizeWindow)
            l21 = np.linspace(0, sizeWindow - 1, sizeWindow)
            l2 = reversed(l21)
            AA2 = []
            for p, q in zip(l1, l2):
                overlapped = overlapK[int(p), :, int(q)]  # [8,]
                AA2.append(overlapped)
            AA.append(AA2)

        elif (i >= numTraj - 1): #if index is bigger than number of trajectory - 1
            if (numTraj < 10):
                l1 = np.linspace(i - sizeWindow + 1, numTraj - 1, lengthOriginal - i)  #
                l21 = np.linspace(i - numTraj + 1, sizeWindow - 1, lengthOriginal - i)
            else:
                l1 = np.linspace(i - sizeWindow, numTraj - 1, lengthOriginal - i)  #
                l21 = np.linspace(i - numTraj, sizeWindow - 1, lengthOriginal - i)
            l2 = reversed(l21)
            AA3 = []
            for p, q in zip(l1, l2):
                overlapped = overlapK[int(p), :, int(q)]
                AA3.append(overlapped)
            AA.append(AA3)
    meanL = []
    for i, e in enumerate(AA):
        arrays = [np.array(x) for x in e]
        meanTime = [np.mean(k) for k in zip(*arrays)]
        meanL.append(meanTime)
    meanL = np.array(meanL)
    return (meanL)

def trainRNN2MSE(sizeWindow,relu,siteNum, tol, model, n_epochs, lrRnn, train_loader, test_loader, batchSizeTrain,
                      batchSizeTest, seq_lengthInput, seq_lengthLabel, inputSize, outputSize):
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lrRnn)

    # Training Run
    epochList = []
    lossListTrain = []
    lossListTest = []
    for epoch in range(1, n_epochs + 1):
        start = time.time()
        epochList.append(epoch)
        lossSumTrain = 0

        #train start
        for batch_idx, (input, label) in enumerate(train_loader):
            input = input.view(batchSizeTrain, seq_lengthInput,
                               inputSize).float()
            label = label.view(batchSizeTrain, seq_lengthLabel,
                               outputSize).float()

            output, h_state = model(input)
            output = output.float()
            lossTrain = criterion(output, label)
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            lossTrain.backward()  # Does backpropagation and calculates gradients
            optimizer.step()
            lossSumTrain = lossSumTrain + lossTrain.detach().numpy()
        lossTrainAvgForOneEpoch1 = lossSumTrain / len(train_loader.dataset)
        lossListTrain.append(lossTrainAvgForOneEpoch1)

        # test start
        with torch.no_grad():
            lossSumTest = 0
            for batch_idx2, (inputTest, labelTest) in enumerate(test_loader):
                inputTest = inputTest.view(batchSizeTest, seq_lengthInput, inputSize).float()
                labelTest = labelTest.view(batchSizeTest, seq_lengthLabel, outputSize).float()
                # Forward pass only to get logits/output
                outputTest, h_stateTest = model(inputTest)
                lossTest = criterion(outputTest, labelTest)
                lossSumTest = lossSumTest + lossTest.detach().numpy()
            lossTestAvgForOneEpoch1 = lossSumTest / len(test_loader.dataset)
            lossListTest.append(lossTestAvgForOneEpoch1)

        end = time.time()

        if lossTrainAvgForOneEpoch1 < tol: #stopping criteria
            break

        if epoch % 1 == 0: #print errors every epoch
            print("Epoch :", epoch, '// One epoch time:', end - start, '// Train Error :', lossTrainAvgForOneEpoch1,
                  '// Test Error :', lossTestAvgForOneEpoch1)

    # Moving Average Method : calculate average of the multiple predictions for each time step
    outputTest = outputTest.detach().numpy()
    outputTest2 = outputTest.reshape((outputTest.shape[0], 1, -1))
    lengthOriginal = 3600
    numTraj = outputTest2.shape[0]
    outputTestOverlapped = calculateOverlap_array(outputTest2, lengthOriginal, sizeWindow, numTraj)

    # save the averaged-prediction for testing period
    np.savetxt(
        './out/predictedTest_epoch{}_batchSize{}_tol{}_siteNum{}_relu{}.csv'.format(n_epochs, batchSizeTrain, tol,
                                                                                       siteNum, relu),
        outputTestOverlapped, delimiter=",")
    return (lossListTrain, lossListTest)


def main():

    siteNum = 2228000 #choose site number you want
    relu= True # add ReLu activation or not
    n_epochs = 1  # number of epoch --> apply about 100k

    mat1 = scipy.io.loadmat('./data/mopex438_daily_10recent.mat')  # dict:5
    data = mat1['dataGauge10']

    # get row number & tolerance (for stopping criteria) for each site number
    rowNum, tol = getRowNumTol(siteNum)

    input, label, d3 = getInputLabel(data, rowNum)
    # np.savetxt('./true{}.csv'.format(siteNum), label, delimiter=",")

    # set window size to apply Moving Average Method
    numDay = 30 #window size

    # create trainInput, trainLabel, testInput, testLabel with windows by shifted by one day
    (trainInputL, trainLabelL, testInputL, testLabelL) = miniBatchDayShiftMonthMovingWinTrainMovingWinTest(input, label, numDay)
    batchSizeTrain = len(trainInputL)
    batchSizeTest = len(testInputL)

    time_step = numDay
    input_size = time_step
    seqLengthInput = 2
    seqLengthLabel = 2
    hidden_size = 32
    nLayer = 4 #number of RNN
    output_size = int(input_size / 2)

    lrRnnAdam = 0.001

    train_xy = getData(trainInputL, trainLabelL)
    test_xy = getData(testInputL, testLabelL)

    train_loader = torch.utils.data.DataLoader(dataset=train_xy,
                                               batch_size=batchSizeTrain,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_xy,
                                              batch_size=batchSizeTest,
                                              shuffle=False)

    model = RNN2(input_size=input_size, output_size=output_size, hidden_dim=hidden_size, n_layers=nLayer)

    trainRNN2MSE(numDay, relu,siteNum, tol, model, n_epochs, lrRnnAdam,
                                                       train_loader, test_loader, batchSizeTrain, batchSizeTest,
                                                       seqLengthInput, seqLengthLabel, input_size, output_size)


if __name__ == "__main__":
    # execute only if run as a script
    main()



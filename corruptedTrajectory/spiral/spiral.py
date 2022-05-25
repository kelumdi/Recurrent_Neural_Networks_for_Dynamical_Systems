
import torch.nn as nn
import torch
import torch.utils.data
import numpy as np
import pandas as pd

np.random.seed(0)
#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def saveFiles(dict, savePath):
    df = pd.DataFrame.from_dict(dict, orient='index').transpose()
    df.to_csv(savePath, index=False)

#convert [numOfAgent,timeStep,2] to [2*numOfAgent,timeStep]
def ten_201_2_TO_20_201_numpy_short(arr):
    arr2 = arr.reshape([arr.shape[0] * arr.shape[2], arr.shape[1]])
    return (arr2)

# referred from : https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
class RNN2(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN2, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining RNN layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True) #input=[batchSize,seqLen,inputSize]

        # Fully connected layer, output=[seq_len, batch_size, num_features]
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

#convert [2*numOfAgent,timeStep] to [numOfAgent,timeStep,2]
def twenty_201_TO_10_201_2_array(twoDim):
    threeDim = []
    for i in range(0, (twoDim.shape[0] - 1)):  # i = 0,...,18
        if i % 2 == 0:
            twoCol = np.transpose(twoDim[i:i + 2, :])
            threeDim.append(twoCol)
    return (np.array(threeDim))

# prepare input & label pairs for train and test data
def trainData(data, label, timeStep):

    # convert [2*numOfAgent,timeStep] to [numOfAgent,timeStep,2]
    y1L = twenty_201_TO_10_201_2_array(data)
    y2L = twenty_201_TO_10_201_2_array(label)
    y1List = np.array([], dtype=np.float64).reshape(0, 2)
    y2List = np.array([], dtype=np.float64).reshape(0, 2)
    for i, e in enumerate(y1L):
        y1List = np.concatenate((y1List, e), axis=0)
    for i, e in enumerate(y2L):
        y2List = np.concatenate((y2List, e), axis=0)

    #make a pair of (input, label) for each agent
    y1L5 = [torch.transpose(torch.tensor(y1List[x:x + timeStep]), 0, 1) for x in
            range(0, len(y1List), timeStep)]
    y2L5 = [torch.transpose(torch.tensor(y2List[x:x + timeStep]), 0, 1) for x in range(0, len(y2List), timeStep)]
    twoList = [(e1, e2) for i, e1 in enumerate(y1L5) for j, e2 in enumerate(y2L5) if
               i == j]
    return (y1L5, y2L5, twoList)

# prepare input & label for train & test data
def getData(data, label, timeStep):
    (y1, y2, y12) = trainData(data, label, timeStep)
    NumDataOneTimeStep = len(y12)
    return (y12, NumDataOneTimeStep)

# train and test
def trainRNN2MSE(sigma,model, n_epochs, lrRnn, train_loader,test_loader, batchSizeTrain, batchSizeTest,seq_length, inputSize,agentTotal):

    criterion = nn.MSELoss() # Define Loss : mean squared loss

    optimizer = torch.optim.Adam(model.parameters(), lr=lrRnn) #define adam optimizer for backpropagation

    # Training Run
    epochList=[]
    lossListTrain = []
    lossListTest=[]
    for epoch in range(1, n_epochs + 1):
        epochList.append(epoch)
        lossSumTrain = 0
        for batch_idx, (input, label) in enumerate(train_loader):

            #change to [batchSizeTrain, seq_length, inputSize]
            input = input.view(batchSizeTrain, seq_length, inputSize).float()
            label = label.view(batchSizeTrain, seq_length, inputSize).float()

            outputTrain, h_state = model(input)  #output of train input
            outputTrain = outputTrain.float()
            lossTrain = criterion(outputTrain,label) #loss
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            lossTrain.backward()  # Does backpropagation and calculates gradients
            optimizer.step()
            # save the current training information
            lossSumTrain = lossSumTrain + lossTrain.detach().numpy()

        with torch.no_grad():
            lossSumTest = 0
            for batch_idx2, (inputTest, labelTest) in enumerate(test_loader):
                inputTest = inputTest.view(batchSizeTest, seq_length, inputSize).float()
                labelTest = labelTest.view(batchSizeTest, seq_length, inputSize).float()

                # Forward pass only to get logits/output
                outputTest, h_stateTest = model(inputTest)
                lossTest = criterion(outputTest, labelTest)
                lossSumTest = lossSumTest + lossTest.detach().numpy()

        lossTrainAvgForOneEpoch1 = lossSumTrain / len(train_loader.dataset)
        lossListTrain.append(lossTrainAvgForOneEpoch1)

        lossTestAvgForOneEpoch1 = lossSumTest / len(test_loader.dataset)#
        lossListTest.append(lossTestAvgForOneEpoch1)

        # save the updated loss and the updated predicted trajectory, automatically
        if epoch==1:
            # convert [numOfAgent,timeStep,2] to [2*numOfAgent,timeStep]
            outputTrain20 = ten_201_2_TO_20_201_numpy_short(outputTrain.detach().numpy())
            outputTest10 = ten_201_2_TO_20_201_numpy_short(outputTest.detach().numpy())
            print("Epoch :", epoch, 'Train Err :', lossTrainAvgForOneEpoch1, 'Test Err :', lossTestAvgForOneEpoch1)

        elif epoch > 1:
            minLossTest = min(lossListTest[0:(epoch - 1)])
            if minLossTest > lossTestAvgForOneEpoch1:
                outputTrain20 = ten_201_2_TO_20_201_numpy_short(outputTrain.detach().numpy())
                outputTest10 = ten_201_2_TO_20_201_numpy_short(outputTest.detach().numpy())
                print("Epoch :", epoch, 'Train Err :', lossTrainAvgForOneEpoch1, 'Test Err :', lossTestAvgForOneEpoch1)
            else:
                print("Epoch :", epoch, ": Current loss is bigger than previous")
                continue
    return (outputTrain20,outputTest10) #output of predicted train and predicted test

def main():

    mu = 0 #mean in gaussian noise
    sigma = 0.4 #standard deviation in gaussian noise
    n_epoch= 2 #number of epoch

    agentTotal=30 #total number of agents

    if agentTotal==30:
        agentTrain = 24  #number of train traj
        agentTest = 6 #number of test traj

    batchSizeTrain = agentTrain #batch size of train data
    batchSizeTest = agentTest #batch size of test data

    if agentTotal == 30:
        pathTrainInput = "./data/30ag/spiral_{}agentTrain_{}agentTotal_noised_{}_{}.csv".format(agentTrain, agentTotal,mu, sigma)
        inputTrain = pd.read_csv(pathTrainInput, sep=',', header=None).to_numpy()

        pathTrainLabel = "./data/30ag/spiral_{}agentTrain_{}agentTotal_true.csv".format(agentTrain,agentTotal)
        labelTrain = pd.read_csv(pathTrainLabel, sep=',', header=None).to_numpy()

        pathTestInput = "./data/30ag/spiral_{}agentTest_{}agentTotal_noised_{}_{}.csv".format(agentTest, agentTotal,mu, sigma)
        inputTest = pd.read_csv(pathTestInput, sep=',', header=None).to_numpy()

        pathTestLabel = "./data/30ag/spiral_{}agentTest_{}agentTotal_true.csv".format(agentTest,agentTotal)
        labelTest = pd.read_csv(pathTestLabel, sep=',', header=None).to_numpy()

    #define parameters : input size, seq length, output size, hidden size, number of RNN, learning rate
    time_step = inputTrain.shape[1]
    seqLength = time_step
    input_size = 2
    hidden_size = 256
    nLayer = 3
    output_size = input_size
    lrRnn = 0.001

    #prepare train data and test data
    (train_xy, NumDataOneTimeStep) = getData(inputTrain, labelTrain, time_step)
    (test_xy, NumDataOneTimeStep) = getData(inputTest, labelTest, time_step)

    train_loader = torch.utils.data.DataLoader(dataset=train_xy,
                                               batch_size=batchSizeTrain,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_xy,
                                               batch_size=batchSizeTest,
                                               shuffle=False)

    #define RNN
    model = RNN2(input_size=input_size, output_size=output_size, hidden_dim=hidden_size, n_layers=nLayer)

    #start train
    (outputTrain20,outputTest10) = trainRNN2MSE(sigma,model, n_epoch, lrRnn, train_loader,test_loader, batchSizeTrain, batchSizeTest,seqLength, input_size,agentTotal)

    ##save predicted train and predicted test
    np.savetxt('./out/spiralPredict_{}agentTrain_{}agentTotal_noised_{}.csv'.format(agentTrain, agentTotal,sigma), outputTrain20, delimiter=",")
    np.savetxt('./out/spiralPredict_{}agentTest_{}agentTotal_noised_{}.csv'.format(agentTest,agentTotal,sigma), outputTest10, delimiter=",")


if __name__ == "__main__":
    # execute only if run as a script
    main()



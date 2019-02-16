#####################################################################################################################
#   CS 6375.003 - Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class NeuralNet:
    def __init__(self, train,split, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        raw_input = pd.read_csv(train)
        # TODO: Remember to implement the preprocess method
        train_dataset = self.preprocess(raw_input,split)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        elif activation == "tanh":
            self.__tanh(self, x)
        elif activation == "relu":
            self.__relu(self,x)

    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        elif activation == "tanh":
            self.__tanh_derivative(self, x)
        elif activation == "relu":
            self.__relu_derivative(self, x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanh(self,x):
        return np.tanh(x)

    def __relu(self,x):
        return x * (x > 0)

    # derivative of sigmoid function, indicates confidence about existing weight
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh_derivative(self,x):
        return 1 - np.square(np.tanh(x))

    def __relu_derivative(self,x):
        return 1 * (x > 0)


    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #
    def preprocess(self, X,split):
        names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
        buying = ['vhigh', 'high', 'med', 'low']
        maint = ['vhigh', 'high', 'med', 'low']
        lug_boot = ['small', 'med', 'big']
        safety = ['low', 'med', 'high']
        cl = ['unacc', 'acc', 'good', 'vgood']
        doors = [2,3,4,5]
        persons = [2,4,5]
        dataset = pd.read_csv(url, names=names)
        dataset = pd.DataFrame(dataset)
        dataset["buying"] = dataset["buying"].astype(pd.Categorical(values=buying, categories=buying)).cat.codes
        dataset["maint"] = dataset["maint"].astype(pd.Categorical(values=maint, categories=maint)).cat.codes
        dataset["lug_boot"] = dataset["lug_boot"].astype(pd.Categorical(values=lug_boot, categories=lug_boot)).cat.codes
        dataset["safety"] = dataset["safety"].astype(pd.Categorical(values=safety, categories=safety)).cat.codes
        dataset["class"] = dataset["class"].astype(pd.Categorical(values=cl, categories=cl)).cat.codes

        dataset["doors"] = dataset["doors"].replace('5more', 5)
        dataset["persons"] = dataset["persons"].replace('more', 5);


        dataset_scaled = preprocessing.scale(dataset)
        dataset_scaled = pd.DataFrame(dataset_scaled)
        pd.DataFrame(dataset_scaled).to_csv('preProcessedCarDataset.csv');

        train,test = train_test_split(dataset_scaled,test_size=split)
       # train = pd.DataFrame(data=train[1:,1:],index=train[1:,0],columns=train[0,1:])
        #test = pd.DataFrame(data=test[1:, 1:], index=test[1:, 0], columns=test[0, 1:])

        train.to_csv('train.csv',sep=',',index=False,header=False)
        test.to_csv('test.csv',sep=',',index=False,header=False)

        return train



    # Below is the training function

    def train(self,activation, max_iterations = 1000, learning_rate = 0.05):
        for iteration in range(max_iterations):
            out = self.forward_pass()
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation=activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("Activation used:"+activation)
        print("Learning Rate:" +str(learning_rate))
        print("No. of nodes in hidden layer 1:" +str(hnode1))
        print("No.of nodes in hidden layer 2:" + str(hnode2))
        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)/len(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self):
        # pass our inputs through our neural network
        in1 = np.dot(self.X, self.w01 )

        if activation == "sigmoid":
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)

        elif activation == "relu":
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)

        elif activation == "tanh":
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)

        return out

    def forward_pass_test(self,test,activation,h1,h2):
        # pass our inputs through our neural network
        coltest=len(test.columns)
        rowtest=len(test.index)
        Xvalues=test.iloc[:,0:(coltest-1)].values.reshape(rowtest,coltest-1)
        Yvalues = test.iloc[:, (coltest - 1)].values.reshape(rowtest,1)

        self.X01=Xvalues
        self.X12=np.zeros((len(Xvalues),h1))
        self.X23=np.zeros((len(Xvalues),h2))
        in1 = np.dot(Xvalues, self.w01 )

        if activation == "sigmoid":
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)

        elif activation == "relu":
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)

        elif activation == "tanh":
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)

        error=0.5 * np.power((out-Yvalues),2)

        return error


    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions

    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))

        self.deltaOut = delta_output

    # TODO: Implement other activation functions

    def compute_hidden_layer2_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        elif activation == "relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))

        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))

        self.delta12 = delta_hidden_layer1

    # TODO: Implement other activation functions

    def compute_input_layer_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "relu":
            delta_input_layer = np.multiply(self.__relu_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))


        self.delta01 = delta_input_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, test,activation,h1,h2, header = True):
        testdata=pd.read_csv(test)
        error=self.forward_pass_test(testdata,activation,h1,h2)
        errorp=np.sum(error)/len(error)

        print("Activation :"+activation)
        print("Test error is:"+str(errorp))
        return 0


if __name__ == "__main__":
    iterations = 1000
    hnode1 = 4
    hnode2 = 2
    activation = "sigmoid"
    learning_rate = 0.05
    neural_network = NeuralNet("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",split=0.3,h1=hnode1,h2=hnode2)
    neural_network.train(learning_rate=learning_rate,activation=activation,max_iterations=iterations)
    testError = neural_network.predict("test.csv",activation,hnode1,hnode2)
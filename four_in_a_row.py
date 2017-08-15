'''
July 2017
@author: Niv Vosco
'''

import numpy as np
import scipy.io as sio

#--------------------------------------------------------------------------------------------------------------------------#

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def randInitializeWeights(in_layer_size, out_layer_size):
    return (np.random.rand(out_layer_size, in_layer_size + 1) * 2 * epsilon - epsilon)

def getParametersOfLayer(theta, layer):
    m,n = X.shape
    ptr_first = 0
    ptr_last = (hidden_layer_size * (input_layer_size + 1))
    a = np.hstack((np.ones((X.shape[0], 1)), X))
    theta1 = (theta[ptr_first:ptr_last]).reshape(hidden_layer_size, input_layer_size + 1)
    z = np.matmul(a ,theta1.T)
    for i in range(0,layer):
        ptr_first = ptr_last
        if (i < num_hidden_layers):
            ptr_last += (layer_size[i+2] * (layer_size[i+1] + 1))
        thetal = (theta[ptr_first:ptr_last]).reshape(layer_size[i+2], layer_size[i+1] + 1)
        a = np.hstack((np.ones((z.shape[0], 1)), sigmoid(z)))
        z = np.matmul(a ,thetal.T)
    return [a,z]

def forwardPropogation(theta, X):
    m,n = X.shape
    ptr_first = 0
    ptr_last = (hidden_layer_size * (input_layer_size + 1))
    z = X
    a = np.hstack((np.ones((z.shape[0], 1)), z))
    for i in range(0,num_hidden_layers+1):
        thetal = (theta[ptr_first:ptr_last]).reshape(layer_size[i+1], layer_size[i] + 1)
        ptr_first = ptr_last
        if (i < num_hidden_layers):
            ptr_last += (layer_size[i+2] * (layer_size[i+1] + 1))
        z = np.matmul(a ,thetal.T)
        a = np.hstack((np.ones((z.shape[0], 1)), sigmoid(z)))
    return a[:,1:]

def gradientDescent(theta, X, y):
    m,n = X.shape
    print("Start: gradient descent")
    for k in range(0, iter_grad):
        if (k % (iter_grad / 10) == 0):
            print("Iteration %d" % k)
        deltal = np.zeros((m, num_labels))
        h = forwardPropogation(theta, X)
        for i in range(0,m):
            for j in range(0, num_labels):
                if (y.item(i,0) == j):
                    deltal[i,j] = h[i,j] - 1
                else:
                    deltal[i,j] = h[i,j]
        [a,z] = getParametersOfLayer(theta, num_hidden_layers)
        theta_grad = (np.matmul(a.T, deltal) / m).T
        theta_grad_t = (np.delete(theta_grad.T, 0, 0)).T
        theta_grad_t = np.hstack((np.zeros((theta_grad_t.shape[0], 1)), theta_grad_t))
        theta_grad = theta_grad + (lambda_reg / m) * theta_grad_t
        ptr_last = theta.shape[0]
        ptr_first = 0
        for i in range(0,num_hidden_layers):
            ptr_first = ptr_first + (layer_size[i+1] * (layer_size[i] + 1))
        for i in range(1,num_hidden_layers+1):
            thetal = (theta[ptr_first:ptr_last]).reshape(layer_size[num_hidden_layers + 2 - i], layer_size[num_hidden_layers + 1 - i] + 1)
            ptr_last = ptr_first
            ptr_first = 0
            for j in range(0,num_hidden_layers-i):
                ptr_first = ptr_first + (layer_size[j+1] * (layer_size[j] + 1))
            [a,z] = getParametersOfLayer(theta, num_hidden_layers - i)
            z = np.hstack((np.ones((z.shape[0], 1)), z))
            delta = np.multiply(np.matmul(deltal, thetal), (np.multiply(sigmoid(z), (1 - sigmoid(z)))))
            delta = (np.delete(delta.T, 0, 0)).T
            theta_grad_t = (np.matmul(a.T, delta) / m).T
            theta_grad_tt = (np.delete(theta_grad_t.T, 0, 0)).T
            theta_grad_tt = np.hstack((np.zeros((theta_grad_tt.shape[0], 1)), theta_grad_tt))
            theta_grad_t = theta_grad_t + (lambda_reg / m) * theta_grad_tt
            theta_grad = (np.append(theta_grad_t.reshape(-1,1), theta_grad.reshape(-1,1))).reshape(-1,1)
            deltal = delta
        theta = theta - (alpha * theta_grad)
    print("Iteration %d" % iter_grad)
    print("Done: gradient descent")
    return theta

def vectorizePredict(theta, X):
    m,n = X.shape
    pred = np.zeros((m, 1))
    h = forwardPropogation(theta, X)
    for i in range(0,m):
        h_i = ((h[i,:]).T).reshape(-1,1)
        pred[i] = ((h_i.argmax(axis=0))[0])
    return pred

def predict(theta, x, allowed):
    h = forwardPropogation(theta, x)
    while (True):
        pred = int(h.argmax())
        if (allowed[pred] == 1):
            break
        h[pred] = 0
    return pred

#--------------------------------------------------------------------------------------------------------------------------#

def clearBoard():
    global board
    board = np.ones((num_of_row, num_of_col))

def isRowWon(row, player):
    for i in range(0, num_of_col - 3):
        if (board[row,i]==player and board[row,i+1]==player and board[row,i+2]==player and board[row,i+3]==player):
            return True
    return False

def isColWon(col, player):
    for i in range(0, num_of_row - 3):
        if (board[i,col]==player and board[i+1,col]==player and board[i+2,col]==player and board[i+3,col]==player):
            return True
    return False

def isDiagWon(player):
    for i in range(0, num_of_row-3):
        for j in range(0, num_of_col-3):
            if (board[i,j]==player and board[i+1,j+1]==player and board[i+2,j+2]==player and board[i+3,j+3]==player):
                return True
            if (board[num_of_row-1-i,num_of_col-1-j]==player and board[num_of_row-2-i,num_of_col-2-j]==player and board[num_of_row-3-i,num_of_col-3-j]==player and board[num_of_row-4-i,num_of_col-4-j]==player):
                return True
    return False

def isPlayerWon(player):
    for i in range(0, num_of_row):
        if (isRowWon(i,player)):
            return True
    for i in range(0, num_of_col):
        if (isColWon(i,player)):
            return True
    if (isDiagWon(player)):
        return True
    return False

def isFullBoard():
    for i in range(0,num_of_col):
        if (board[num_of_row-1,i]==empty_slot):
            return False
    return True

def isGameOver():
    if (isPlayerWon(player1) or isPlayerWon(player2) or isFullBoard()):
        return True
    return False

def findNextRow(col):
    for i in range(0, num_of_row):
        if (board[i,col]==empty_slot):
            return i
    return (-1)

def getAllowedPlays():
    p = np.zeros((num_of_col,1))
    for i in range(0, num_of_col):
        if (board[num_of_row-1,i]==empty_slot):
            p[i] = 1
    return p

def playerMove(theta, player):
    global board
    x = (board.reshape(-1,1)).T
    mov_col = predict(theta, x, getAllowedPlays())
    mov_row = findNextRow(mov_col)
    board[mov_row, mov_col] = player

def playRandomMove(player):
    global board
    row = -1
    while (row < 0):
        col = np.random.randint(0, num_of_col)
        row = findNextRow(col)
    board[row, col] = player
    return col

def printBoard():
    print_board = board - empty_slot
    for i in range(num_of_row-1,-1,-1):
        for j in range(0,num_of_col):
            print("%d " % (print_board[i,j]),end='')
        print ('')

def getNewTrainingSample(last_board, last_move):
    x = (last_board.reshape(-1,1)).T
    y = last_move
    return [x,y]

def isWinPrevented(last_move, opp, player):
    global board
    row = findNextRow(last_move)
    if (row < 0):
        board[num_of_row-1,last_move] = opp
        win = isPlayerWon(opp)
        board[num_of_row-1,last_move] = player
    else:
        board[row-1,last_move] = opp
        win = isPlayerWon(opp)
        board[row-1,last_move] = player
    return win

def calculateAccuracy(predictions):
    m,n = X.shape
    temp = predictions - y
    return (100 * ((np.count_nonzero(temp==0)) / m))

def gatherTrainingExamples():
    X = np.zeros((training_set, input_layer_size))
    y = np.zeros((training_set, 1))
    training_set_count = 0
    trainig_done = False
    print("Start: gather training examples")
    print("Training example %d" % training_set_count)
    while (training_set_count < training_set):
        clearBoard()
        while (training_set_count < training_set):
            last_move = playRandomMove(player1)
            if(isGameOver()):
                break
            last_board = board
            last_move = playRandomMove(player2)
            if(isGameOver()):
                break
            if(isWinPrevented(last_move, player1, player2)):
                [X[training_set_count,:], y[training_set_count,:]] = getNewTrainingSample(last_board, last_move)
                training_set_count += 1
                if (training_set_count % (training_set / 10) == 0):
                    print("Training example %d" % training_set_count)
        if (training_set_count >= training_set):
            break
        if (isPlayerWon(player2)):
            [X[training_set_count,:], y[training_set_count,:]] = getNewTrainingSample(last_board, last_move)
            training_set_count += 1
            if (training_set_count % (training_set / 10) == 0):
                print("Training example %d" % training_set_count)
    print("Done: gather training examples")
    return [X,y]

#--------------------------------------------------------------------------------------------------------------------------#


## Steup the game parameters
num_of_col = 7
num_of_row = 6
board = np.ones((num_of_row, num_of_col))
empty_slot = 1
player1 = 2
player2 = 3

## Setup the neural network parameters
input_layer_size  = num_of_col * num_of_row
num_hidden_layers = 2
hidden_layer_size = int(input_layer_size / 2)
num_labels = num_of_col
epsilon = 0.12           # Range for parameter initialization
lambda_reg = 0.5         # Regularization parameter
iter_grad = 1000000      # Number of iterations to run gradient descent
training_set = 100000    # Training set size
alpha = 1                # Learning rate for gradient descent
layer_size = [0] * (num_hidden_layers + 2)
layer_size[0] = input_layer_size
layer_size[num_hidden_layers + 1] = num_labels
for i in range(1,num_hidden_layers+1):
    layer_size[i] = hidden_layer_size

## Inialize the parameters
initial_nn_params = np.zeros((0,0))
for i in range(0,num_hidden_layers + 1):
    initial_theta = randInitializeWeights(layer_size[i], layer_size[i+1])
    initial_nn_params = (np.append(initial_nn_params, initial_theta.reshape(-1,1))).reshape(-1,1)

char_var = input("Press t to train the network or any other key to start a new game: ")
if (char_var == 't'):

    ## Gather the training set (randomize play)
    [X,y] = gatherTrainingExamples()

    ## Run gradient descent (train the neural network)
    theta = gradientDescent(initial_nn_params, X, y)
    sio.savemat('theta.mat', {'theta':theta})

    ## Check accuaracy of the model
    pred = vectorizePredict(theta, X)
    print("\nTraining Set Accuracy: ", calculateAccuracy(pred))
else:
    try:
        theta = sio.loadmat('theta.mat')["theta"]
    except:
        theta = initial_nn_params

## Play against the user
while (True):
    char_var = input("Press q to quit or any other key to start a new game: ")
    if (char_var == 'q'):
        break
    clearBoard()
    while (not isGameOver()):
        printBoard()
        input_var = 0
        while (input_var<1 or input_var>num_of_col):
            try:
                input_var = int(input("Please enter the number of column(1 - %d): " % (num_of_col)))
            except:
                continue
        plyaer_row = findNextRow(input_var-1)
        board[plyaer_row,input_var-1] = player1
        if (isGameOver()):
            break
        playerMove(theta, player2)
    printBoard()
    if (isPlayerWon(player1)):
        print("You won")
    if (isPlayerWon(player2)):
        print("You lost")

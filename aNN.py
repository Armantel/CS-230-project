import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# Helper functions
def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    """
    Implements the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0,Z)

    assert(A.shape == Z.shape)

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    Implements the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)

    return dZ




# Initialize weight distribution
def initialize_w_distribution (w_centered, w_range):

    '''
    This creates a truncated normal distribution of w either around the mean of w_range (option 1) or
    with mean = 0 (option 2) to emulate the initial distribution of weights in the semiconductor device array
    '''

    if w_centered == False:
        lower, upper = w_range[0,0], w_range[0,1]
        mu, sigma = np.mean(w_range), (upper - np.mean(w_range)) / (w_range[0,1] - w_range[0,0]) / 1e2
        # Last division term (for sigma) is arbitrary and can be measured experimentally in the future.

        w_distribution = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc = mu, scale = sigma)

    elif w_centered == True:
        lower, upper = -w_range[0,1], w_range[0,1]
        mu, sigma = np.mean([lower, upper]), (upper - np.mean([lower, upper])) / (upper - lower) / 1e2
        # Last division term (for sigma) is arbitrary and can be measured experimentally in the future.

        w_distribution = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc = mu, scale = sigma)


    # Plot distribution
    '''
    N = stats.norm(loc = mu, scale = sigma)
    fig, ax = plt.subplots(2, sharex = True)
    ax[0].hist(w_distribution.rvs(10000), normed=True)
    ax[1].hist(w_distribution.rvs(10000), normed=True)
    plt.xlim(w_range[0,0],w_range[0,1])
    plt.show()
    '''


    return w_distribution




# Initialize parameters W and b
def initialize_parameters_deep(layers_dims, w_limit, w_range, w_distribution, w_centered):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in the network

    Returns:
    parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(1)
    parameters = {}
    L = len(layers_dims)     # number of layers in the network

    for l in range(1, L):

        # Xavier initialization
        if w_limit[1] == "Xavier":
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) / np.sqrt(layers_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))


        elif w_limit[1] != "Xavier":
            '''
            Below makes some of the weights negative using multiplication by -1
            The code is rather awkward and can be simplified in a future version
            '''

            ind_w = np.random.random((layers_dims[l], layers_dims[l-1]))   # Draw uniform random number from [0,1)
            for i in range(ind_w.shape[0]):
                for j in range(ind_w.shape[1]):
                    if ind_w[i,j] <0.5:
                        ind_w[i,j] = -1
                    else:
                        ind_w[i,j] = 1

            ind_b = np.random.random((layers_dims[l], 1)) # Draw uniform random number from [0,1)
            for i in range(ind_b.shape[0]):
                for j in range(ind_b.shape[1]):
                    if ind_b[i,j] <0.5:
                        ind_b[i,j] = -1
                    else:
                        ind_b[i,j] = 1


            if w_limit[1] == "Uniform":

                '''
                Below populates the weights following a uniform distribution
                truncated by w_limit
                '''
                parameters['W' + str(l)] = np.random.uniform(w_range[0, 0], w_range[0, 1], (layers_dims[l], layers_dims[l-1]))
                parameters['b' + str(l)] = np.random.uniform(w_range[0, 0], w_range[0, 1], (layers_dims[l], 1))

                parameters['W' + str(l)] = parameters['W' + str(l)] * ind_w     # Multiply some of the weights by -1
                parameters['W' + str(l)] = parameters['W' + str(l)] * ind_b


            elif w_limit[1] == "Normal":

                '''
                Below populates the weights following a normal distribution
                truncated by w_limit, see initialize_w_distribution function
                '''
                parameters['W' + str(l)] = w_distribution.rvs((layers_dims[l], layers_dims[l-1]))   # / np.sqrt(layers_dims[l-1])
                parameters['b' + str(l)] = w_distribution.rvs(((layers_dims[l], 1)))

                if w_centered == False:
                    parameters['W' + str(l)] = parameters['W' + str(l)] * ind_w     # Multiply some of the weights by -1
                    parameters['W' + str(l)] = parameters['W' + str(l)] * ind_b


        assert(parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layers_dims[l], 1))

    return parameters




# Forward propagation (computes Z)
def linear_forward(A, W, b):
    """
    Implements the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = W.dot(A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache




# Forward propagation (Computes A)
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implements forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)  # cache = (A, W, b)
        A, activation_cache = relu(Z)                   # cache = Z

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache




# Forward propagation (L-1 times ReLU, 1 time Sigmoid)
def L_model_forward(X, parameters):
    """
    Implements forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A

        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation = "relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation = "sigmoid")
    caches.append(cache)


    assert(AL.shape == (1,X.shape[1]))

    return AL, caches




# Compute cost
def compute_cost(AL, Y):
    """
    Implements the Logistic regression cost function.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())

    return cost




# Linear backward (Computes dA_prev, dW, db)
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db




# Linear backward + activation (calculates dA_prev, dW, db)
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db




# L times linear backward + activation (calculates grads)
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) #Following logistic regression

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")

    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads




# Update parameters
def update_parameters(parameters, grads, learning_rate, noise, w_limit, w_range):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter.
    for l in range(L):

        if noise == 0:
            parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)] # + noise/100 * np.random.uniform(0, 1) * grads["dW" + str(l+1)])
            parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)] # + noise/100 * np.random.uniform(0, 1) * grads["db" + str(l+1)])


        # Noise represents the % value of the total range
        # When noise is present the weights are updated such that the variance between successive updates is added
        elif noise != 0:

            '''
            TBD
            dW_range_fraction = []
            db_range_fraction = []

            dW_range_fraction = np.abs(learning_rate * (grads["dW" + str(l+1)] / (w_range[0,1] - w_range[0,0])))
            db_range_fraction = np.abs(learning_rate * (grads["db" + str(l+1)] / (w_range[0,1] - w_range[0,0])))

            assert(dW_range_fraction.shape == grads["dW" + str(l+1)].shape)
            assert(db_range_fraction.shape == grads["db" + str(l+1)].shape)

            # Factor of 2 comes in because p(g2T-TT) ENODe updates are ~0.5% of the total range, and the noise of ~40% (or chosen value) is with respect to that number
            parameters["W" + str(l+1)] -= learning_rate * (grads["dW" + str(l+1)] + np.random.normal(0, np.sqrt(dW_range_fraction * 2 * (noise/100)**2)) * grads["dW" + str(l+1)])
            parameters["b" + str(l+1)] -= learning_rate * (grads["db" + str(l+1)] + np.random.normal(0, np.sqrt(db_range_fraction * 2 * (noise/100)**2)) * grads["db" + str(l+1)])
            '''

            # Uniform error
            parameters["W" + str(l+1)] -= learning_rate * (grads["dW" + str(l+1)] + np.random.uniform(-1, np.ones(grads["dW" + str(l+1)].shape)) * noise * grads["dW" + str(l+1)])
            parameters["b" + str(l+1)] -= learning_rate * (grads["db" + str(l+1)] + np.random.uniform(-1, np.ones(grads["db" + str(l+1)].shape)) * noise * grads["db" + str(l+1)])


        if w_limit[0] == True:
            '''
            Below clips the weights and bias units to those allowed by w_range
            '''

            # Positive weights
            parameters["W" + str(l+1)][(parameters["W" + str(l+1)] > 0) & (parameters["W" + str(l+1)] < w_range[0,0])] = w_range[0,0]
            parameters["W" + str(l+1)][(parameters["W" + str(l+1)] > 0) & (parameters["W" + str(l+1)] > w_range[0,1])] = w_range[0,1]

            # Negative weights
            parameters["W" + str(l+1)][(parameters["W" + str(l+1)] < 0) & (parameters["W" + str(l+1)] > -w_range[0,0])] = -w_range[0,0]
            parameters["W" + str(l+1)][(parameters["W" + str(l+1)] < 0) & (parameters["W" + str(l+1)] < -w_range[0,1])] = -w_range[0,1]

            # Positive bias
            parameters["b" + str(l+1)][(parameters["b" + str(l+1)] > 0) & (parameters["b" + str(l+1)] < w_range[0,0])] = w_range[0,0]
            parameters["b" + str(l+1)][(parameters["b" + str(l+1)] > 0) & (parameters["b" + str(l+1)] > w_range[0,1])] = w_range[0,1]

            # Negative bias
            parameters["b" + str(l+1)][(parameters["b" + str(l+1)] < 0) & (parameters["b" + str(l+1)] > -w_range[0,0])] = -w_range[0,0]
            parameters["b" + str(l+1)][(parameters["b" + str(l+1)] < 0) & (parameters["b" + str(l+1)] < -w_range[0,1])] = -w_range[0,1]



    return parameters




# L layer model
def L_layer_model(X, Y, X_test, Y_test, layers_dims, learning_rate, num_iterations, print_cost, noise, w_limit, w_centered, w_range):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                   # keeps track of cost
    accuracy_train = []          # keeps track of accuracy
    accuracy_test = []

    # Parameters initialization
    parameters = initialize_parameters_deep(layers_dims, w_limit, w_range, w_distribution = initialize_w_distribution (w_centered, w_range), w_centered = w_centered)
    L = len(parameters) // 2        # number of layers in the neural network


    # This is awkward (!), but works for now.
    import copy
    parameters_copy = copy.deepcopy(parameters)


    # Save initial weight(w) and bias(b) distributions
    w_initial = {}

    for l in range(L):
        w_initial["W" + str(l+1)] = parameters_copy["W" + str(l+1)].reshape(parameters_copy["W" + str(l+1)].shape[0] * parameters_copy["W" + str(l+1)].shape[1], -1)
        w_initial["b" + str(l+1)] = parameters_copy["b" + str(l+1)].reshape(parameters_copy["b" + str(l+1)].shape[0] * parameters_copy["b" + str(l+1)].shape[1], -1)



    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
        AL, caches = L_model_forward(X, parameters)

        # Compute cost
        cost = compute_cost(AL, Y)

        # Backward propagation
        grads = L_model_backward(AL, Y, caches)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate, noise, w_limit, w_range)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

            # Training accuracy
            accuracy_train.append(predict(X = X, Y = Y, parameters = parameters))
            print("Training ccuracy: %s" % str(predict(X = X, Y = Y, parameters = parameters)))

            # Test accuracy
            accuracy_test.append(predict(X = X_test, Y = Y_test, parameters = parameters))
            print("Test accuracy: %s" % str(predict(X = X_test, Y = Y_test, parameters = parameters)))


    # Plot the cost
    fig1 = plt.figure()
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('100x iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # Plot accuracy
    fig2 = plt.figure()
    plt.plot(accuracy_train)
    plt.plot(accuracy_test)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('100x iterations')
    plt.legend(['Training accuracy', 'Test accuracy'])
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # Save final weight(w) and bias(b) distributions
    w_final = {}

    # Plot initial/final weight(b) and bias(b) distributions
    fig3, axs = plt.subplots(L, L)


    for l in range(L):
        w_final["W" + str(l+1)] = parameters["W" + str(l+1)].reshape(parameters["W" + str(l+1)].shape[0] * parameters["W" + str(l+1)].shape[1], -1)
        w_final["b" + str(l+1)] = parameters["b" + str(l+1)].reshape(parameters["b" + str(l+1)].shape[0] * parameters["b" + str(l+1)].shape[1], -1)

        # Weights (w)
        axs[0,l].set_xlabel("w" + str(l))
        axs[0,l].set_ylabel("Counts")
        #axs[0,l].set_xlim(-1, 1)
        axs[0,l].hist(w_initial["W" + str(l+1)], bins = 100, alpha = 0.5)
        axs[0,l].hist(w_final["W" + str(l+1)], bins = 100, alpha = 0.5)

        # Bias (b)
        axs[1,l].set_xlabel("b" + str(l))
        axs[1,l].set_ylabel("Counts")
        #axs[1,l].set_xlim(-1, 1)
        axs[1,l].hist(w_initial["b" + str(l+1)], alpha = 0.5)
        axs[1,l].hist(w_final["b" + str(l+1)], alpha = 0.5)

    axs[0,1].legend(["Initial", "Final"])
    plt.tight_layout()


    return parameters, caches, w_initial, w_final, accuracy_train, accuracy_test, costs




# Check model accuracy
def predict(X, Y, parameters):
    """
    This function is used to predict the results of an L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    predictions = np.zeros((1, m),dtype=int)

    # Forward propagation
    probabilities, caches = L_model_forward(X, parameters)


    # Convert probabilities to 0/1 predictions
    for i in range(0, probabilities.shape[1]):
        if probabilities[0,i] > 0.5:
            predictions[0,i] = 1
        else:
            predictions[0,i] = 0

    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(Y))

    accuracy = np.sum(predictions == Y)/float(m) * 100
    # print("Accuracy: %s" % str(accuracy))

    return accuracy

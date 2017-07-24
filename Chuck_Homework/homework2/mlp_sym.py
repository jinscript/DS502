import mxnet as mx


def mlp_layer(input_layer, n_hidden, activation=None, BN=False):

    """
    A MLP layer with activation layer and BN
    :param input_layer: input sym
    :param n_hidden: # of hidden neurons
    :param activation: the activation function
    :return: the symbol as output
    """

    # get a FC layer
    l = mx.sym.FullyConnected(data=input_layer, num_hidden=n_hidden)
    # get activation, it can be relu, sigmoid, tanh, softrelu or none
    if activation is not None:
        l = mx.sym.Activation(data=l, act_type=activation)
    if BN:
        l = mx.sym.BatchNorm(l)
    return l


def get_mlp_sym():

    """
    :return: the mlp symbol
    """

    data = mx.sym.Variable("data")
    # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
    data_f = mx.sym.flatten(data=data)

    # Your Design
    l = mlp_layer(input_layer=data_f, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)

    # MNIST has 10 classes
    l = mx.sym.FullyConnected(data=l, num_hidden=10)
    # Softmax with cross entropy loss
    mlp = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return mlp


def conv_layer(input_layer,
               conv_kernel=(1, 1),
               conv_num_filter=10,
               if_pool=False,
               pool_kernel=(1, 1),
               pool_stride=(1, 1)):
    """
    :return: a single convolution layer symbol
    """
    # todo: Design the simplest convolution layer
    # Find the doc of mx.sym.Convolution by help command
    # Do you need BatchNorm?
    # Do you need pooling?
    # What is the expected output shape?
    l = mx.sym.Convolution(data=input_layer, kernel=conv_kernel, num_filter=conv_num_filter)
    l = mx.sym.Activation(data=l, act_type='relu')
    l = mx.sym.BatchNorm(data=l)

    if if_pool == True:
        l = mx.sym.Pooling(data=l, pool_type='max', kernel=pool_kernel, stride=pool_stride)
    return l

# Optional
def inception_layer():
    """
    Implement the inception layer in week3 class
    :return: the symbol of a inception layer
    """
    pass


def get_conv_sym():

    """
    :return: symbol of a convolutional neural network
    """
    data = mx.sym.Variable("data")
    # todo: design the CNN architecture
    # How deep the network do you want? like 4 or 5
    # How wide the network do you want? like 32/64/128 kernels per layer
    # How is the convolution like? Normal CNN? Inception Module? VGG like?

    data = mx.sym.Variable("data")

    l = conv_layer(input_layer=data, conv_kernel=(5, 5), conv_num_filter=20, if_pool=True, pool_kernel=(2, 2), pool_stride=(2, 2))
    l = conv_layer(input_layer=l, conv_kernel=(5, 5), conv_num_filter=50, if_pool=True, pool_kernel=(2, 2), pool_stride=(2, 2))
    l = mx.sym.flatten(data=l)
    l = mlp_layer(l, 500, activation='relu', BN=True)
    l = mlp_layer(l, 10)
    cnn = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return cnn

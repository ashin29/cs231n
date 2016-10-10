import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    pad = (filter_size - 1) / 2
    stride = 1
    out_h = ((H + 2*pad - filter_size) / stride + 1) / 2
    out_w = ((W + 2*pad - filter_size) / stride + 1) / 2
    
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = weight_scale * np.random.randn(num_filters*out_h*out_w, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out, cache_1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    out, cache_2 = affine_relu_forward(out, W2, b2)
    scores, cache_scores = affine_forward(out, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    reg = self.reg
    loss, d_out = softmax_loss(scores, y)
    loss += 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
    
    d_out, d_W3, d_b3 = affine_backward(d_out, cache_scores)
    d_out, d_W2, d_b2 = affine_relu_backward(d_out, cache_2)
    d_x, d_W1, d_b1 = conv_relu_pool_backward(d_out, cache_1)
    
    grads['W1'] = d_W1 + reg * W1
    grads['b1'] = d_b1
    grads['W2'] = d_W2 + reg * W2
    grads['b2'] = d_b2
    grads['W3'] = d_W3 + reg * W3
    grads['b3'] = d_b3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  

class MultiLayerConvNet(object):
  """
  A multi-layer convolutional neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {conv-[bn]-relu--conv-[bn]-relu-pool}xL - affine - softmax
  
  where batch normalization and dropout are optional.
  
  Learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, num_layers=1, input_dim=(3, 32, 32), num_classes=10,
               conv_stride=1, conv_pad=1, conv_num_filters=32, conv_filter_size=7,
               pool_height=2, pool_width=2, pool_stride=2,
               dropout=0, use_batchnorm=True, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new MultiLayerConvNet.
    
    Inputs:
    - num_layers: An integer giving the number of layers.
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_classes: An integer giving the number of classes to classify.
    
    - conv_stride: The number of pixels between adjacent receptive fields in the
                   horizontal and vertical directions.
    - conv_pad: The number of pixels that will be used to zero-pad the input.
    - conv_num_filters: Number of filters to use in the convolutional layer
    - conv_filter_size: Size of filters to use in the convolutional layer
      
    - pool_height: The height of each pooling region
    - pool_width: The width of each pooling region
    - pool_stride: The distance between adjacent pooling regions  
      
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
               the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
                    initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
             this datatype. float32 is faster but less accurate, so you should use
             float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
            will make the dropout layers deteriminstic so we can gradient check the
            model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = num_layers
    self.conv_param = {'stride': conv_stride, 
                       'pad': conv_pad, 
                       'num_filters': conv_num_filters,
                       'filter_size': conv_filter_size}
    self.pool_param = {'pool_height': pool_height, 'pool_width': pool_width, 'stride': pool_stride}
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    C, H, W = input_dim
    
    out_c = C
    out_h = H
    out_w = W
    for i in xrange(1, self.num_layers+1):
        in_c = out_c
        in_h = out_h
        in_w = out_w
        conv1_out_h = (in_h + 2*conv_pad - conv_filter_size) / conv_stride + 1
        conv1_out_w = (in_w + 2*conv_pad - conv_filter_size) / conv_stride + 1
        conv2_out_h = (conv1_out_h + 2*conv_pad - conv_filter_size) / conv_stride + 1
        conv2_out_w = (conv1_out_w + 2*conv_pad - conv_filter_size) / conv_stride + 1
        pool_out_h = (conv2_out_h - pool_height) / pool_stride + 1
        pool_out_w = (conv2_out_w - pool_width) / pool_stride + 1
        out_c = conv_num_filters
        out_h = pool_out_h
        out_w = pool_out_w
    
        self.params['conv1_W%d' % (i)] = weight_scale * np.random.randn(conv_num_filters, in_c, conv_filter_size, conv_filter_size)
        self.params['conv1_b%d' % (i)] = np.zeros(conv_num_filters)
        self.params['conv2_W%d' % (i)] = weight_scale * np.random.randn(conv_num_filters, conv_num_filters, conv_filter_size, conv_filter_size)
        self.params['conv2_b%d' % (i)] = np.zeros(conv_num_filters)
        
        if self.use_batchnorm:
            self.params['bn1_gamma%d' % (i)] = np.ones(conv_num_filters)
            self.params['bn1_beta%d' % (i)] = np.zeros(conv_num_filters)
            self.params['bn2_gamma%d' % (i)] = np.ones(conv_num_filters)
            self.params['bn2_beta%d' % (i)] = np.zeros(conv_num_filters)
    
    self.params['W_score'] = weight_scale * np.random.randn(out_c*out_h*out_w, num_classes)
    self.params['b_score'] = np.zeros(num_classes)
        
    #for k, v in self.params.iteritems():
    #    print k, v.shape
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn1_params = []
    self.bn2_params = []
    if self.use_batchnorm:
      self.bn1_params = [{'mode': 'train'} for i in xrange(self.num_layers)]
      self.bn2_params = [{'mode': 'train'} for i in xrange(self.num_layers)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn1_param in self.bn1_params:
        bn1_param[mode] = mode
      for bn2_param in self.bn2_params:
        bn2_param[mode] = mode  

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    caches_1 = {}
    caches_2 = {}
    out = X
    for i in xrange(1, self.num_layers+1):
        if self.use_batchnorm:
            out, caches_1[i] = conv_bn_relu_forward(out,
                                                    self.params['conv1_W%d' % (i)], 
                                                    self.params['conv1_b%d' % (i)],
                                                    self.conv_param,
                                                    self.params['bn1_gamma%d' % (i)], 
                                                    self.params['bn1_beta%d' % (i)],
                                                    self.bn1_params[i-1])
            out, caches_2[i] = conv_bn_relu_pool_forward(out,
                                                         self.params['conv2_W%d' % (i)], 
                                                         self.params['conv2_b%d' % (i)],
                                                         self.conv_param,
                                                         self.params['bn2_gamma%d' % (i)], 
                                                         self.params['bn2_beta%d' % (i)],
                                                         self.bn2_params[i-1],
                                                         self.pool_param)                                          
        else:
            out, caches_1[i] = conv_relu_forward(out, 
                                                 self.params['conv1_W%d' % (i)], 
                                                 self.params['conv1_b%d' % (i)],
                                                 self.conv_param)
            out, caches_2[i] = conv_relu_pool_forward(out, 
                                                      self.params['conv2_W%d' % (i)], 
                                                      self.params['conv2_b%d' % (i)],
                                                      self.conv_param,
                                                      self.pool_param)                                     
        
    scores, cache_scores = affine_forward(out, 
                                          self.params['W_score'], 
                                          self.params['b_score'])
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, d_out = softmax_loss(scores, y)
    for i in xrange(1, self.num_layers+1):
        loss += 0.5 * self.reg * np.sum(self.params['conv1_W%d' % (i)]**2)
        loss += 0.5 * self.reg * np.sum(self.params['conv2_W%d' % (i)]**2)
        
    loss += 0.5 * self.reg * np.sum(self.params['W_score']**2)

    d_x = d_out
    d_x, grads['W_score'], grads['b_score'] = affine_backward(d_x, cache_scores)
    grads['W_score'] += self.reg * self.params['W_score']
    
    for i in reversed(xrange(1, self.num_layers+1)):
        if self.use_batchnorm:
            d_x, \
            grads['conv2_W%d' % (i)], grads['conv2_b%d' % (i)], \
            grads['bn2_gamma%d' % (i)], grads['bn2_beta%d' % (i)] = conv_bn_relu_pool_backward(d_x, caches_2[i])
            
            d_x, \
            grads['conv1_W%d' % (i)], grads['conv1_b%d' % (i)], \
            grads['bn1_gamma%d' % (i)], grads['bn1_beta%d' % (i)] = conv_bn_relu_backward(d_x, caches_1[i])
        else:
            d_x, grads['conv2_W%d' % (i)], grads['conv2_b%d' % (i)] = conv_relu_pool_backward(d_x, caches_2[i])
            d_x, grads['conv1_W%d' % (i)], grads['conv1_b%d' % (i)] = conv_relu_backward(d_x, caches_1[i])
        
        grads['conv1_W%d' % (i)] += self.reg * self.params['conv1_W%d' % (i)]
        grads['conv2_W%d' % (i)] += self.reg * self.params['conv2_W%d' % (i)]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

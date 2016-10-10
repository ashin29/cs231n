from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that performs affine-bn-relu.
    
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta, bn_param: Parameters for the BatchNorm layer
    
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    fc_out, fc_cache = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(fc_out, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn_out)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache
    
    
def affine_bn_relu_backward(dout, cache):
    """
    Backward pass for the affine-bn-relu convenience layer
    """
    (fc_cache, bn_cache, relu_cache) = cache
    d_bn_out = relu_backward(dout, relu_cache)
    d_fc_out, d_gamma, d_beta = batchnorm_backward(d_bn_out, bn_cache)
    d_x, d_w, d_b = affine_backward(d_fc_out, fc_cache)
    return d_x, d_w, d_b, d_gamma, d_beta
    

def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

  
def conv_bn_relu_forward(x, w, b, conv_param, gamma, beta, bn_param):
  """
  A convenience layer that performs conv-bn-relu.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - gamma, beta, bn_param: Parameters for the BatchNorm layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  out_conv, cache_conv = conv_forward_fast(x, w, b, conv_param)
  out_bn, cache_bn = spatial_batchnorm_forward(out_conv, gamma, beta, bn_param)
  out, cache_relu = relu_forward(out_bn)
  cache = (cache_conv, cache_bn, cache_relu)
  return out, cache


def conv_bn_relu_backward(dout, cache):
  """
  Backward pass for the conv-bn-relu convenience layer.
  """
  cache_conv, cache_bn, cache_relu = cache
  d_out_bn = relu_backward(dout, cache_relu)
  d_out_conv, d_gamma, d_beta = spatial_batchnorm_backward(d_out_bn, cache_bn)
  dx, dw, db = conv_backward_fast(d_out_conv, cache_conv)
  return dx, dw, db, d_gamma, d_beta
  

def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_bn_relu_pool_forward(x, w, b, conv_param, gamma, beta, bn_param, pool_param):
  """
  A convenience layer that performs conv-bn-relu-pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - gamma, beta, bn_param: Parameters for the BatchNorm layer
  - pool_param: Parameters for the pooling layer
  
  Returns a tuple of:
  - out: Output from the pooling
  - cache: Object to give to the backward pass
  """
  out_conv, cache_conv = conv_forward_fast(x, w, b, conv_param)
  out_bn, cache_bn = spatial_batchnorm_forward(out_conv, gamma, beta, bn_param)
  out_relu, cache_relu = relu_forward(out_bn)
  out, cache_pool = max_pool_forward_fast(out_relu, pool_param)
  cache = (cache_conv, cache_bn, cache_relu, cache_pool)
  return out, cache


def conv_bn_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-bn-relu-pool convenience layer.
  """
  cache_conv, cache_bn, cache_relu, cache_pool = cache
  d_out_pool = max_pool_backward_fast(dout, cache_pool)
  d_out_bn = relu_backward(d_out_pool, cache_relu)
  d_out_conv, d_gamma, d_beta = spatial_batchnorm_backward(d_out_bn, cache_bn)
  dx, dw, db = conv_backward_fast(d_out_conv, cache_conv)
  return dx, dw, db, d_gamma, d_beta
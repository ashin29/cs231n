import numpy as np
import math
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  
  # compute the loss and the gradient
  for i in xrange(num_train):
    scores = X[i, :].dot(W)  # (1, C)
    scores -= np.max(scores)  # normalization trick, for numeric stability
    
    sum_exp_score = 0
    for j in xrange(num_class):
        sum_exp_score += np.exp(scores[j])
        
    exp_correct_score = np.exp(scores[y[i]])
    normalized_prob = exp_correct_score / sum_exp_score
    loss += -math.log(normalized_prob)
    
    for j in xrange(num_class):
        dW[:, j] += np.exp(scores[j]) * X[i, :].T / sum_exp_score
        
        if j == y[i]:
            dW[:, j] -= X[i, :]
  
  
  # Averaging
  loss /= num_train
  dW /= num_train
  
  # Add regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  
  # compute the loss and the gradient
  scores = X.dot(W)  # (N, C)
  scores -= np.amax(scores, axis=1, keepdims=True)  # (N, C)
  exp_scores = np.exp(scores)  # (N, C)
  
  sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)  # (N, 1)
  exp_correct_scores = exp_scores[range(num_train), y].reshape(-1, 1)  # (N, 1)
  normalized_correct_probs = exp_correct_scores / sum_exp_scores  # (N, 1)
  losses = -np.log(normalized_correct_probs)  # (N, 1)
  loss = sum(losses)
  
  normalized_probs = exp_scores / sum_exp_scores  # (N, C)
  normalized_probs[range(num_train), y] -= 1
  dW = np.dot(X.T, normalized_probs)
  # Averaging
  loss /= num_train
  dW /= num_train
  
  # Add regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


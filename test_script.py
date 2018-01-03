from network import *
import util
import numpy as np

(training, validation, test) = util.load_data_wrapper();
net = Network(3, [784, 30, 10])
for i in range(10):
  print "training"
  net.train(training[0], training[1], 3.0, 10, 3)
  print "testing"
  print net.n_correct(test[0], test[1])

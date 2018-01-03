import sys
sys.path.insert(0, "/home/lselker/Documents/Personal/Developing/nielson/neural-networks-and-deep-learning-master/src")

import network
import numpy as np
import time
net = network.Network([784, 30, 10])
input = np.random.rand(784, 1)
truth = np.random.rand(10, 1)
start = time.clock()
for i in range(1):
  net.backprop(input, truth)
end = time.clock()
print "1 backprops:", (end - start)

e = np.random.rand(10, 1)
a = np.random.rand(1, 784)
start = time.clock()
m = []
for i in range(3):
  m.append( np.dot(e, a))
end = time.clock()
print "3 multiplications: ", (end - start)
inp = np.random.rand(30)
start = time.clock()
network.sigmoid_prime(inp)
end = time.clock()
print "af_prime: ", end - start

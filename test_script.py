from network import *
net = Network(3, [3, 2, 1])
print net.feed_forward([1, 0, 1])
print type(net.feed_forward([1, 0, 1]))
for i in range(100):
  net.train([[1, 0, 1]], [[0.5]], 0.5, 1, 1)
  if (i % 10)  == 0:
    print net.feed_forward([1, 0, 1])

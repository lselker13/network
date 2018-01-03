import gzip
import cPickle
import numpy as np

# Formatted data due to http://neuralnetworksanddeeplearning.com
# The data loading functions here are mostly from there as well.

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    """
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` have the same format.
    """
    print "Loading data from file..."
    tr_d, va_d, te_d = load_data()
    print "Done"
    print "Formatting data..."
    training_inputs = [x.tolist() for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]

    validation_inputs = [x.tolist() for x in va_d[0]]
    validation_results = [vectorized_result(y) for y in va_d[1]]

    test_inputs = [x.tolist() for x in te_d[0]]
    test_results = [vectorized_result(y) for y in te_d[1]]
    print "Done"
    return ((training_inputs, training_results),
            (validation_inputs, validation_results),
            (test_inputs, test_results))

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = []
    for i in range(10):
      e.append(1 if i == j else 0)
    return e

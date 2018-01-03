#include<iostream>
#include<Eigen/Dense>
#include<algorithm>
#include<assert.h>
#include<vector>
#include<random>
#include<chrono>

#include"network.h"

// @author: Leo Selker
// A simple backpropogating neural network

using namespace Eigen;
using namespace std;

double Network::sigmoid_simple(const double &x) {
  double xexp = exp(-x);
  return 1/(1 + xexp);
}

double Network::sigmoid_prime_simple(const double &x) {
  return sigmoid_simple(x) * (1 - sigmoid_simple(x));
}

// activation function
MatrixXd Network::af(const MatrixXd &input) {
  return input.unaryExpr(&(Network::sigmoid_simple));
}

MatrixXd Network::af_prime(const MatrixXd &input) {
  return input.unaryExpr(&Network::sigmoid_prime_simple);
}

int Network::i_max(Eigen::VectorXd in) {
  // Returns the lower index in a tie
  if(in.size() == 0) {
    return -1;
  }
  int i = 0;
  double max = in(0);
  for(int j = 0; j < in.size(); j++) {
    if(in(j) > max) {
      i = j;
      max = in(j);
    }
  }
  return i;
}

Network::Network(int n_layers, vector<int> sizes) {
  Network::n_layers = n_layers;
  Network::sizes = sizes;
  Network::weights = vector<MatrixXd>(n_layers - 1);
  Network::biases = vector<VectorXd>(n_layers - 1);
  for(int source_layer = 0; source_layer < n_layers - 1; source_layer++) {
    Network::weights[source_layer] = (MatrixXd::Random(sizes[source_layer + 1], sizes[source_layer]));
    Network::biases[source_layer] = (VectorXd::Random(sizes[source_layer + 1]));
  }
};

Network::Network(int n_layers, vector<int> sizes, vector<MatrixXd> weights, vector<VectorXd> biases) {
  Network::n_layers = n_layers;
  Network::sizes = sizes;
  Network::weights = weights;
  Network::biases = biases;
  }

VectorXd Network::feed_forward(VectorXd a) {
  for(int source_layer = 0; source_layer < n_layers - 1; source_layer++) {
    a = af((Network::weights[source_layer] * a) + Network::biases[source_layer]);
  }
  return a;
}

vector<double> Network::feed_forward(vector<double> a) {
    VectorXd result = feed_forward(to_eigen(a));
    return from_eigen(result);
}

double Network::c_simple(const double &a, const double &y) {
  return pow(a - y, 2);
}

double Network::c_prime_simple(const double &a, const double &y) {
  return 2 * (a - y);
}

double Network::c(const MatrixXd &a, const MatrixXd &y) {
  MatrixXd costs = a.binaryExpr(y, &Network::c_simple);
  return costs.sum();
}

MatrixXd Network::c_prime(const MatrixXd &a, const MatrixXd &y) {
  return a.binaryExpr(y, &(Network::c_prime_simple));
}

// x is the input, y is the expected output
Partials Network::backprop(const VectorXd &x, const VectorXd &y) {
  // Activations starts at layer 0, the input layer
  vector<VectorXd> activations(n_layers);
  // Errors and weighted uputs startt at layer 1, the first layer with weighted input
  vector<VectorXd> errors(n_layers - 1);
  vector<VectorXd> weighted_inputs(n_layers - 1);
  activations[0] = x;
  // Feedforward, populate activations and weighted inputs, initialize errors
  // TODO: Why is this ~5 times slower than the numpy equivalent?
  for (int i = 0; i < n_layers - 1; i++) {
    errors[i] = VectorXd(sizes[i + 1]);
    weighted_inputs[i] = (weights[i] * activations[i]) + biases[i];
    activations[i + 1] = af(weighted_inputs[i]);
  }

  // Compute error for last layer (special case b/c cost function)
  // Vector of partials of activation w/r/t weighted input
  VectorXd p_activation_weighted_input = af_prime(weighted_inputs[n_layers - 2]);
  errors[n_layers - 2] = p_activation_weighted_input.array()
      * c_prime(activations[n_layers - 1], y).array();
  // back-propogate to compute errors, starting w/ second to last layer
  for (int l = n_layers - 3; l >= 0; l--) {
    errors[l] = af_prime(weighted_inputs[l]).array()
        * (weights[l + 1].transpose() * errors[l + 1]).array();
  }

  vector<MatrixXd> weight_partials(n_layers - 1);
  // Compute the weight partials using the errors
  for (int l = 0; l < n_layers - 1; l++) {
    weight_partials[l] = errors[l] * activations[l].transpose();
  }

  // bias partials are equal to errors
  return Partials(weight_partials, errors);
}

// Take a set of inputs and truths, update based on average of partials
// TODO: Use an iterator
void Network::update_batch(
    const vector<VectorXd> &inputs, const vector<VectorXd> &truths,
    vector<int>::iterator start, vector<int>::iterator end, double rate) {

  assert(inputs.size() == truths.size());
  vector<MatrixXd> weight_partials(n_layers - 1);
  vector<VectorXd> bias_partials(n_layers - 1);
  for (int i = 0; i < n_layers - 1; i++) {
    weight_partials[i] = MatrixXd::Zero(sizes[i+1], sizes[i]);
    bias_partials[i] = VectorXd::Zero(sizes[i+1]);
  }
  Partials partials(weight_partials, bias_partials);
  int iterations = 0;
  for(vector<int>::iterator it = start; it != end; it++) {
    iterations++;
    int i = *it;
    assert(i < (int) inputs.size());
    assert(inputs[i].size() == sizes[0]);
    assert(truths[i].size() == sizes[n_layers - 1]);
    Partials d_partials = backprop(inputs[i], truths[i]);
    for(int j = 0; j < n_layers - 1; j++) {
      partials.weight_partials[j] = partials.weight_partials[j] + d_partials.weight_partials[j];
      partials.bias_partials[j] = partials.bias_partials[j] + d_partials.bias_partials[j];
    }
  }
  double adjustment = rate / iterations;
  for(int j = 0; j < n_layers - 1; j++) {
    weights[j] = weights[j] - (partials.weight_partials[j] * adjustment);
    biases[j] = biases[j] - (partials.bias_partials[j] * adjustment);
  }

}

void Network::train(
    vector<VectorXd> inputs, vector<VectorXd> truths,
    double rate, int batch_size, int epochs) {
  vector<int> indices(inputs.size());
  int n_inputs = indices.size();
  for(int i = 0; i < n_inputs; i++) {
    indices[i] = i;
  }
  // TODO: Determine exact behavior/how deterministic this is
  for(int c = 0; c < epochs; c++) {
    random_shuffle(indices.begin(), indices.end());
    int start = 0;
    int end = 0;
    while(start < n_inputs) {
      end = min(n_inputs, start + batch_size);
      update_batch(inputs, truths, indices.begin() + start, indices.begin() + end, rate);
      start = end;
    }
  }
}

void Network::train(vector< vector <double> > inputs,
                    vector<vector<double> > truths, double rate, int batch_size, int epochs){
  train(map_to_eigen(inputs), map_to_eigen(truths), rate, batch_size, epochs);
}

vector<double> Network::test(vector<VectorXd> inputs,vector<VectorXd> truths) {
  vector<double> ret(inputs.size());
  for(int i = 0; i < (int) inputs.size(); i++) {
    assert(inputs[i].size() == sizes[0]);
    assert(truths[i].size() == sizes[n_layers - 1]);
    ret[i] = c(feed_forward(inputs[i]), truths[i]);
  }
  return ret;
}

vector<double> Network::test(vector<vector<double> > inputs, vector<vector<double> > truths) {
  return test(map_to_eigen(inputs), map_to_eigen(truths));
}


template<class T>
Matrix<T, Dynamic, 1> Network::to_eigen(vector<T> in) {
  Map<Eigen::Matrix<T, Dynamic, 1> > ret(in.data(), in.size());
  return ret;
}

template<class T>
vector<T> Network::from_eigen(Matrix<T, Dynamic, 1> in) {
  vector<T> ret(in.data(), in.data() + in.rows() * in.cols());
  return ret;
}

template<class T>
vector<Matrix<T, Dynamic, 1> > Network::map_to_eigen(vector<vector<T> > in) {
  vector<Matrix<T, Dynamic, 1> > ret(in.size());
  for(int i = 0; i < in.size(); i++) {
    ret[i] = to_eigen(in[i]);
  }
  return ret;
}

int Network::n_correct(vector<VectorXd> inputs, vector<VectorXd> truths) {
  int total = 0;
  for(int i = 0; i < (int) inputs.size(); i++) {
    assert(inputs[i].size() == sizes[0]);
    assert(truths[i].size() == sizes[n_layers - 1]);
    int result = i_max(feed_forward(inputs[i]));
    int expected = i_max(truths[i]);
    if(result == expected) {
      total++;
    }
  }
  return total;
}

int Network::n_correct(vector<vector<double> > inputs, vector<vector<double> > truths) {
  return n_correct(map_to_eigen(inputs), map_to_eigen(truths));
}

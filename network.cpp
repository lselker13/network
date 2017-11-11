#include<iostream>
#include<Eigen/Dense>
#include<assert.h>
#include<vector>

#include"network.h"

using namespace Eigen;
using namespace std;

double Network::sigmoid_simple(double x) {
  double xexp = exp(-x);
  return 1/(1 + xexp);
}

double Network::sigmoid_prime_simple(double x) {
  return sigmoid_simple(x) * (1 - sigmoid_simple(x));
}

// activation function
MatrixXd Network::af(MatrixXd input) {
  return input.unaryExpr(&(Network::sigmoid_simple));
}

MatrixXd Network::af_prime(MatrixXd input) {
  return input.unaryExpr(&Network::sigmoid_prime_simple);
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

double Network::c_simple(double a, double y) {
  return pow(a - y, 2);
}

double Network::c_prime_simple(double a, double y) {
  return 2 * (a - y);
}

double Network::c(MatrixXd a, MatrixXd y) {
  MatrixXd costs = a.binaryExpr(y, &Network::c_simple);
  return costs.sum();
}

MatrixXd Network::c_prime(MatrixXd a, MatrixXd y) {
  return a.binaryExpr(y, &(Network::c_prime_simple));
}

// x is the input, y is the expected output
Partials Network::backprop(VectorXd x, VectorXd y) {
  // Activations starts at layer 0, the input layer
  vector<VectorXd> activations(n_layers);
  // Errors and weighted uputs startt at layer 1, the first layer with weighted input
  vector<VectorXd> errors(n_layers - 1);
  vector<VectorXd> weighted_inputs(n_layers - 1);
  activations[0] = x;
  cout << "\n ACTIVATION\n" << activations[0];
  // Feedforward, populate activations and weighted inputs, initialize errors
  for (int i = 0; i < n_layers - 1; i++) {
    errors[i] = VectorXd(sizes[i + 1]);
    weighted_inputs[i] = (weights[i] * activations[i]) + biases[i];
    activations[i + 1] = af(weighted_inputs[i]);
    cout << "\n ACTIVATION\n" << activations[i + 1];
  }

  // Compute error for last layer (special case b/c cost function)
  // Vector of partials of activation w/r/t weighted input
  VectorXd p_activation_weighted_input = af_prime(weighted_inputs[n_layers - 2]);
  errors[n_layers - 2] = p_activation_weighted_input.array()
      * c_prime(activations[n_layers - 1], y).array();
  cout << "\n P_AF\n" << sigmoid_prime_simple(0.5788619707);
  cout << "\n ERROR\n" << errors[n_layers - 2];
  // back-propogate to compute errors, starting w/ second to last layer
  for (int l = n_layers - 3; l >= 0; l--) {
    errors[l] = af_prime(weighted_inputs[l]).array()
        * (weights[l + 1].transpose() * errors[l + 1]).array();
    cout << "\n ERROR \n" << errors[l];
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
void Network::update_batch(
    vector<VectorXd> inputs, vector<VectorXd> truths, int n_inputs, double rate) {
  if(n_inputs == 0) {
    return;
  }
  assert(inputs[0].size() == sizes[0]);
  assert(truths[0].size() == sizes[n_layers - 1]);
  Partials partials = backprop(inputs[0], truths[0]);
  for(int i = 1; i < n_inputs; i++) {
    assert(inputs[i].size() == sizes[0]);
    assert(truths[i].size() == sizes[n_layers - 1]);
    Partials d_partials = backprop(inputs[i], truths[i]);
    for(int j = 0; j < n_layers - 1; j++) {
      partials.weight_partials[j] = partials.weight_partials[j] + d_partials.weight_partials[j];
      partials.bias_partials[j] = partials.bias_partials[j] + d_partials.bias_partials[j];
    }
  }
  double adjustment = rate / n_inputs;

  for(int j = 0; j < n_layers - 1; j++) {
    weights[j] = weights[j] - (partials.weight_partials[j] * adjustment);
    biases[j] = biases[j] - (partials.bias_partials[j] * adjustment);
  }

}

// Does not use batch size yet - passes all inputs along
void Network::train(
    vector<VectorXd> inputs, vector<VectorXd> truths, int n_inputs,
    double rate,int batch_size, int epochs) {
  for(int c = 1; c < epochs; c++) {
    update_batch(inputs, truths, n_inputs, rate);
  }
}

vector<double> Network::test(vector<VectorXd> inputs,vector<VectorXd> truths, int n_inputs, vector<double> ret_dest) {
  for(int i = 0; i < n_inputs; i++) {
    assert(inputs[i].size() == sizes[0]);
    assert(truths[i].size() == sizes[n_layers - 1]);
    ret_dest[i] = c(feed_forward(inputs[i]), truths[i]);
  }
  return ret_dest;
}

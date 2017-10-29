#include<iostream>
#include<Eigen/Dense>

#include"network.h"

using namespace Eigen;

double Network::sigmoid_simple(double x) {
  double xexp = exp(-x);
  return 1/(1 + xexp);
}

double Network::sigmoid_prime_simple(double x) {
  return sigmoid_simple(x) * (1 - sigmoid_simple(x));
}

// activation function
MatrixXd Network::af(MatrixXd input) {
  return input.unaryExpr(&Network::sigmoid_simple);
}

MatrixXd Network::af_prime(MatrixXd input) {
  return input.unaryExpr(&Network::sigmoid_prime_simple);
}

Network::Network(int n_layers, int* sizes) {
  Network::n_layers = n_layers;
  Network::sizes = sizes;
  Network::weights = new MatrixXd[(const int)(n_layers - 1)];
  Network::biases = new VectorXd[(const int)(n_layers - 1)];
  for(int source_layer = 0; source_layer < n_layers - 1; source_layer++) {
    Network::weights[source_layer] = MatrixXd::Random(sizes[source_layer + 1], sizes[source_layer]);
    Network::biases[source_layer] = VectorXd::Random(sizes[source_layer + 1]);
  }
}

Network::Network(int n_layers, int* sizes, MatrixXd* weights, VectorXd* biases) {
  Network::n_layers = n_layers;
  Network::sizes = sizes;
  Network:: weights = weights;
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
  return a.binaryExpr(y, &Network::c_prime_simple);
}


// x is the input, y is the expected output
Partials Network::backprop(VectorXd x, VectorXd y) {
  // Activations starts at layer 0, the input layer
  VectorXd activations[n_layers];
  // Errors and weighted uputs startt at layer 1, the first layer with weighted input
  VectorXd errors[n_layers - 1];
  VectorXd weighted_inputs[n_layers - 1];
  activations[0] = x;

  // Feedforward, populate activations and weighted inputs, initialize errors
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

  MatrixXd weight_partials[n_layers - 1];

  // Compute the weight partials using the errors
  for (int l = 0; l < n_layers - 1; l++) {
    weight_partials[l] = errors[l] * activations[l].transpose();
  }

  // bias partials are equal to errors
  return Partials(weight_partials, errors);
}



int main(void) {
  int n_layers = 3;
  int sizes[3] = {3,4,2};
  Network network(n_layers, sizes);
  Vector3d a(1,2,1);
  VectorXd o = network.feed_forward(a);
  std::cout << o << "\n";

  Partials p;
  MatrixXd m = Matrix3d::Random();
  p.weight_partials = &m;
  VectorXd v = Vector3d::Random();
  p.bias_partials = &v;
  std::cout << "bp \n" << *p.weight_partials;

}

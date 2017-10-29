#ifndef NETWORK_H
#define NETWORK_H

#include<Eigen/Dense>
using namespace Eigen;


class Partials {
 public:
  MatrixXd* weight_partials;
  VectorXd* bias_partials;
  Partials() {}
  Partials(MatrixXd* weight_partials, VectorXd* bias_partials) {
    Partials::weight_partials = weight_partials;
    Partials::bias_partials = bias_partials;
  }
};


class Network {
  int n_layers;
  int* sizes;
  MatrixXd* weights;
  VectorXd* biases;

  // Activation function
  double sigmoid_simple(double x);
  double sigmoid_prime_simple(double x);
  MatrixXd af(MatrixXd input);
  MatrixXd af_prime(MatrixXd input);

  // Cost function
  double c_simple(double a, double y);
  // partial of elementwise cost w/r/t that output activation
  double c_prime_simple(double a, double y);
  double c(MatrixXd a, MatrixXd y);
  // vector of partials of cost function with respect to output activations
  MatrixXd c_prime(MatrixXd a, MatrixXd y);

 public:

  Network(int n_layers, int* sizes);
  Network(int n_layers, int* sizes, MatrixXd* weights, VectorXd* biases);
  VectorXd feed_forward(VectorXd a);
  Partials backprop(VectorXd x, VectorXd y);

};

#endif

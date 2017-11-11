#ifndef NETWORK_H
#define NETWORK_H

#include<Eigen/Dense>
#include<vector>
using namespace Eigen;
using namespace std;


class Partials {
 public:
  vector<MatrixXd> weight_partials;
  vector<VectorXd> bias_partials;
  Partials() {}
  Partials(vector<MatrixXd> weight_partials, vector<VectorXd> bias_partials) {
    Partials::weight_partials = weight_partials;
    Partials::bias_partials = bias_partials;
  }
};

class Network {
  int n_layers;
  vector<int> sizes;
  vector<MatrixXd> weights;
  vector<VectorXd> biases;

  // Activation function
  static double sigmoid_simple(double x);
  static double sigmoid_prime_simple(double x);
  static MatrixXd af(MatrixXd input);
  static MatrixXd af_prime(MatrixXd input);

  // Cost function
  static double c_simple(double a, double y);
  // partial of elementwise cost w/r/t that output activation
  static double c_prime_simple(double a, double y);
  static double c(MatrixXd a, MatrixXd y);
  // vector of partials of cost function with respect to output activations
  static MatrixXd c_prime(MatrixXd a, MatrixXd y);

 public:

  Network(int n_layers, vector<int> sizes);
  Network(int n_layers, vector<int> sizes, vector<MatrixXd> weights, vector<VectorXd> biases);
  VectorXd feed_forward(VectorXd a);
  Partials backprop(VectorXd x, VectorXd y);
  void update_batch(vector<VectorXd> inputs, vector<VectorXd> truths, int n_inputs, double rate);
  void train(vector<VectorXd> inputs, vector<VectorXd> truths, int n_inputs,
             double rate, int batch_size, int epochs);
  vector<double> test(
      vector<VectorXd> inputs, vector<VectorXd> truths, int n_inputs, vector<double> ret_dest);

};

#endif

#ifndef NETWORK_H
#define NETWORK_H

#include<Eigen/Dense>
#include<vector>

// @author: Leo Selker

// To test private members (should not be used outside testing)
namespace unit_test {
struct network_tester;
}

class Partials {
 public:
  std::vector<Eigen::MatrixXd> weight_partials;
  std::vector<Eigen::VectorXd> bias_partials;
  Partials() {}
  Partials(std::vector<Eigen::MatrixXd> weight_partials, std::vector<Eigen::VectorXd> bias_partials) {
    Partials::weight_partials = weight_partials;
    Partials::bias_partials = bias_partials;
  }
};

class Network {
 private:
  int n_layers;
  std::vector<int> sizes;
  std::vector<Eigen::MatrixXd> weights;
  std::vector<Eigen::VectorXd> biases;

  // Activation function
  static double sigmoid_simple(double x);
  static double sigmoid_prime_simple(double x);
  static Eigen::MatrixXd af(Eigen::MatrixXd input);
  static Eigen::MatrixXd af_prime(Eigen::MatrixXd input);

  // Cost function
  static double c_simple(double a, double y);
  // partial of elementwise cost w/r/t that output activation
  static double c_prime_simple(double a, double y);
  static double c(Eigen::MatrixXd a, Eigen::MatrixXd y);
  // vector of partials of cost function with respect to output activations
  static Eigen::MatrixXd c_prime(Eigen::MatrixXd a, Eigen::MatrixXd y);

  Partials backprop(Eigen::VectorXd x, Eigen::VectorXd y);
  void update_batch(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> truths,
                    int n_inputs, double rate);

 public:
  Network(int n_layers, int* sizes);
  Network(int n_layers, std::vector<int> sizes);
  Network(int n_layers, std::vector<int> sizes, std::vector<Eigen::MatrixXd> weights,
          std::vector<Eigen::VectorXd> biases);
  Eigen::VectorXd feed_forward(Eigen::VectorXd a);
  void train(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> truths,
             int n_inputs, double rate, int batch_size, int epochs);
  std::vector<double> test(
      std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> truths,
      int n_inputs, std::vector<double> ret_dest);

  friend struct unit_test::network_tester;
};

#endif

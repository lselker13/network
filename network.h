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
  static double sigmoid_simple(const double &x);
  static double sigmoid_prime_simple(const double &x);
  static Eigen::MatrixXd af(const Eigen::MatrixXd &input);
  static Eigen::MatrixXd af_prime(const Eigen::MatrixXd &input);

  // Cost function
  static double c_simple(const double &a, const double &y);
  // partial of elementwise cost w/r/t that output activation
  static double c_prime_simple(const double &a, const double &y);
  static double c(const Eigen::MatrixXd &a, const Eigen::MatrixXd &y);
  // vector of partials of cost function with respect to output activations
  static Eigen::MatrixXd c_prime(const Eigen::MatrixXd &a, const Eigen::MatrixXd &y);

  Partials backprop(const Eigen::VectorXd &x, const Eigen::VectorXd &y);

  void update_batch(const std::vector<Eigen::VectorXd> &inputs,
                    const std::vector<Eigen::VectorXd> &truths,
                    std::vector<int>::iterator start,
                    std::vector<int>::iterator end, double rate);

  template<class T>
  static Eigen::Matrix<T, Eigen::Dynamic, 1> to_eigen(std::vector<T> in);

  template<class T>
  static std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1> > map_to_eigen(
      std::vector<std::vector<T> > in);

  template<class T>
  static std::vector<T> from_eigen(Eigen::Matrix<T, Eigen::Dynamic, 1> in);

  static int i_max(Eigen::VectorXd in);



 public:

  Network(int n_layers, std::vector<int> sizes);
  Network(int n_layers, std::vector<int> sizes, std::vector<Eigen::MatrixXd> weights,
          std::vector<Eigen::VectorXd> biases);


  Eigen::VectorXd feed_forward(Eigen::VectorXd a);
  std::vector<double> feed_forward(std::vector<double> a);

  void train(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> truths,
             double rate, int batch_size, int epochs);
  void train(std::vector< std::vector <double> > inputs, std::vector< std::vector<double> > truths,
             double rate, int batch_size, int epochs);

  std::vector<double> test(
      std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> truths);
  std::vector<double> test(
      std::vector<std::vector<double> > inputs, std::vector< std::vector<double> > truths);

  int n_correct(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> truths);
  int n_correct(std::vector<std::vector<double> > inputs, std::vector< std::vector<double> > truths);

  friend struct unit_test::network_tester;
};

#endif

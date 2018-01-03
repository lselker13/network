#include"network.h"

#include<memory>
#include<iostream>
#include<assert.h>
#include<Eigen/Dense>
#include<vector>
#include<time.h>

/**
 * A small hand-verified regression test
 * @author: Leo Selker
 */

using namespace Eigen;
using namespace std;
// TODO: Use a testing libary

// Define test friend
namespace unit_test {
struct network_tester {
  static vector<MatrixXd> get_weights(Network net) {
    return net.weights;
  }
  static vector<VectorXd> get_biases(Network net) {
    return net.biases;
  }
  static Partials backprop(Network net, Eigen::VectorXd x, Eigen::VectorXd y) {
    return net.backprop(x, y);
  }
  /*  TODO: Figure out why this doesnt work
  template<class T>
  static Matrix<T, Dynamic, 1> to_eigen(vector<T> in) {
    return Network::to_eigen<T>(in);
  }
  template<class T>
  static vector<Matrix<T, Dynamic, 1> > map_to_eigen(vector<vector<T> > in) {
    return Network::map_to_eigen<T>(in);
  }
  template<class T>
  static vector<T> from_eigen(Matrix<T, Dynamic, 1> in) {
    return Network::from_eigen<T>(in);
    }*/
};
}

std::unique_ptr<Network> network;

int n_layers = 3;
vector<int> sizes = {3, 2, 1};
vector<MatrixXd> weights(2);
vector<VectorXd> biases(2);

Partials expected_partials;

// TODO: Eliminate the need for this arbitrary value
const double EPSILON = pow(10, -5);

Vector3d input(1, 0, 1);
double expected = 0.578862;


void before() {
  weights[0] = Matrix<double, 2, 3>();
  weights[0] << 1, 1, 0, 0, 0, 2;
  weights[1] = Matrix<double, 1, 2>();
  weights[1] << 0.5, 1;

  biases[0] = Vector2d(0,1);
  biases[0] << 0, 1;
  biases[1] = VectorXd(1);
  biases[1] << -1;

  network.reset(new Network(n_layers, sizes, weights, biases));

  /**
   * Expected partials calculated at
   * https://docs.google.com/spreadsheets/d/1lhi0bSxK6XB0UcGM49ebSvdQIsgMAmlQDvZepGrECtg/
   * edit?usp=sharing
   */

  vector<MatrixXd> weight_partials(2);
  vector<VectorXd> bias_partials(2);
  weight_partials[0] = Matrix<double, 2, 3>();
  weight_partials[0] << 0.003779871001, 0, 0.003779871001, 0.001737045593, 0, 0.001737045593;
  weight_partials[1] = Matrix<double, 1, 2>();
  weight_partials[1] << 0.02810925131, 0.03662653898;
  bias_partials[0] = Vector2d();
  bias_partials[0] << 0.003779871001, 0.001737045593;
  bias_partials[1] = VectorXd(1);
  bias_partials[1] << 0.03845006698;;
  expected_partials = Partials(weight_partials, bias_partials);

}
/*
void test_to_eigen() {
  vector<int> in({1, 2, 3, 4, 5});
  VectorXi exp(5);
  exp << 1, 2, 3, 4, 5;
  VectorXi result = unit_test::network_tester::to_eigen(in);
  assert(result.isApprox(exp));
}

void test_from_eigen() {
  VectorXi in(5);
  in << 1, 2, 3, 4, 5;
  vector<int> exp({1, 2, 3, 4, 5});
  vector<int> result = unit_test::network_tester::from_eigen(in);
  assert(exp == result);
}
*/
void test_feed_forward() {
  VectorXd output = network -> feed_forward(input);
  assert(std::abs(expected - output(0)) < EPSILON);
}

void test_backprop() {
  VectorXd truth = VectorXd(1);
  truth << 0.5;
  Partials partials = unit_test::network_tester::backprop(*network, input, truth);
  for(int i = 0; i < n_layers - 1; i++) {
    assert(partials.weight_partials[i].isApprox(expected_partials.weight_partials[i], EPSILON));
    assert(partials.bias_partials[i].isApprox(expected_partials.bias_partials[i], EPSILON));
  }
}

void test_train() {
  // Run a single piece of training data. Expectation is that the weights and biases increment
  // by the expected partials.
  vector<VectorXd> inputs = {input};
  VectorXd truth(1);
  truth << 0.5;
  vector<VectorXd> truths = {truth};
  double rate = 1;
  int batch_size = 1; // Doesn't matter currently
  int epochs = 1;
  network -> train(inputs, truths, rate, batch_size, epochs);

  // Pull out the biases and weights using the test friend
  vector<MatrixXd> new_weights = unit_test::network_tester::get_weights(*network);
  vector<VectorXd> new_biases = unit_test::network_tester::get_biases(*network);


  for(int i = 0; i < n_layers - 1; i++) {
    MatrixXd expected_weights = weights[i] - expected_partials.weight_partials[i];
    MatrixXd expected_biases = biases[i] - expected_partials.bias_partials[i];
    assert(new_weights[i].isApprox(expected_weights, EPSILON));
    assert(new_biases[i].isApprox(expected_biases, EPSILON));
  }
}

void benchmark() {
  vector<int> sizes = {784, 30, 10};
  Network net(3, sizes);
  VectorXd x = VectorXd::Random(784);
  VectorXd y = VectorXd::Random(10);
  time_t start = time(NULL);
  for(int i = 0; i < 10000; i++) {
    unit_test::network_tester::backprop(net, x, y);
  }
  time_t end = time(NULL);
  cout << "Time to backprop 10000 times: " << difftime(start, end);
}

int main(void) {
  before();
  test_feed_forward();
  test_backprop();
  test_train();
  benchmark();
  cout << "PASS\n";
}

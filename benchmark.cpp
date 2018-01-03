#include"network.h"

#include<memory>
#include<iostream>
#include<Eigen/Dense>
#include<vector>
#include<time.h>
#include<chrono>

/**
 * Finding bottlenecks
 * @author: Leo Selker
 */

using namespace Eigen;
using namespace std;

// We use the same friend to get more granularity by calling private methods
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
  static MatrixXd af_prime(MatrixXd input) {
    return Network::af_prime(input);
  }
};
}

unique_ptr<Network> network;

int n_layers = 3;
vector<int> sizes = {784, 30, 10};


void before() {

  network.reset(new Network(n_layers, sizes));

}

void af_prime_time() {
  VectorXd in = VectorXd::Random(30);
  auto start = std::chrono::high_resolution_clock::now();
  unit_test::network_tester::af_prime(in);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
  cout << " Time for af_prime: " << duration/pow(10, 9) << "\n";
}


void multiplication_time() {
  // Figure out how to allocate memory well
  MatrixXd e = VectorXd::Random(10);
  VectorXd a = VectorXd::Random(784);
  MatrixXd r = e * a.transpose();
  MatrixXd* p = &r;
  auto start = chrono::high_resolution_clock::now();
  for(int i = 0; i < 2; i++) {
    e * a.transpose();
  }
  auto end = chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
  cout << " Time to multiply 3 times: " << duration/pow(10, 9);
}

void backprop_time() {
  VectorXd x = VectorXd::Random(784);
  VectorXd y = VectorXd::Random(10);
  auto start = chrono::high_resolution_clock::now();
  for(int i = 0; i < 1; i++) {
    unit_test::network_tester::backprop(*network, x, y);
  }
  auto end = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::nanoseconds>(end-start).count();
  cout << "Time to backprop 100 times: " << duration/pow(10, 9);
}

int main(void) {
  before();

  af_prime_time();
  multiplication_time();
  backprop_time();
}

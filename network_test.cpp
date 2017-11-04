#include"network.h"

#include<memory>
#include<iostream>
#include<assert.h>
#include<Eigen/Dense>

using namespace Eigen;

Network* network;
int n_layers = 3;
int sizes[3] = {3, 2, 1};
MatrixXd weights[2];
VectorXd biases[2];

void before() {


  weights[0] = Matrix<double, 2, 3>();
  weights[1] = Matrix<double, 1, 2>();
  weights[0] << 1, 1, 0, 0, 0, 2;
  weights[1] << 0.5, 1;

  biases[0] = Vector2d(0,1);
  biases[1] = VectorXd(1);
  biases[0] << 0, 1;
  biases[1] << -1;

  network = new Network(n_layers, sizes, weights, biases);

}

void after() {
  delete network;
}

void test_feed_forward() {
  Vector3d input(1, 0, 1);
  VectorXd output = network -> feed_forward(input);
  std::cout << output;
}

int main(void) {
  before();
  test_feed_forward();
  after();
}

#include"network.h"

#include<memory>
#include<iostream>
#include<assert.h>
#include<Eigen/Dense>
#include<vector>

using namespace Eigen;
using namespace std;
// TODO: Use a testing libary

std::unique_ptr<Network> network;

int n_layers = 3;
vector<int> sizes = {3, 2, 1};
vector<MatrixXd> weights(2);
vector<VectorXd> biases(2);
double EPSILON = pow(10, -5);
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

}


void test_feed_forward() {
  VectorXd output = network -> feed_forward(input);
  assert(std::abs(expected - output(0)) < EPSILON);
}

void test_backprop() {
  // Expected partials calculated at
  //https://docs.google.com/spreadsheets/d/1lhi0bSxK6XB0UcGM49ebSvdQIsgMAmlQDvZepGrECtg/edit?usp=sharing
  vector<MatrixXd> weight_partials(2);
  vector<VectorXd> bias_partials(2);
  weight_partials[0] = Matrix<double, 2, 3>();
  weight_partials[0] << 0.00199092392, 0, 0.00199092392, 0.003646347771, 0, 0.003646347771;
  weight_partials[1] = Matrix<double, 1, 2>();
  weight_partials[1] << 0.01327014628, 0.01729108771;
  bias_partials[0] = Vector2d();
  bias_partials[0] << 0.00199092392, 0.003646347771;
  bias_partials[1] = VectorXd(1);
  bias_partials[1] << 0.01815196027;
  Partials expected_partials(weight_partials, bias_partials);
  VectorXd truth = VectorXd(1);
  truth << 0.5;
  Partials partials = network -> backprop(input, truth);
  for(int i = 0; i < 2; i++) {
    cout << "\n WEIGHT \n" << partials.weight_partials[i];
    cout << "\n BIAS \n" << partials.bias_partials[i];
    // assert(partials.weight_partials[i].isApprox(expected_partials.weight_partials[i]));
    // assert(partials.bias_partials[i].isApprox(expected_partials.bias_partials[i]));
  }
}

int main(void) {
  before();
  test_feed_forward();
  test_backprop();
}

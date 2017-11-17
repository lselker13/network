%module network
%{
  #include "network.h"
  #include<vector>
%}

%include "std_vector.i"

namespace std {
  %template(v1i)  vector < int >;
  %template(v2i) vector < vector < int> >;
  %template(v1d) vector < double >;
  %template(v2d) vector < vector < double > >;
}

Class Network {
public:
  Network(int n_layers, std:vector<int> sizes);
  
  std::vector<double> feed_forward(std::vector<double> a);
  
  void train(std::vector< std::vector <double> > inputs, std::vector< std::vector<double> > truths,
             int n_inputs, double rate, int batch_size, int epochs);
  
  std::vector<double> test(
      std::vector<std::vector<double> > inputs, std::vector< std::vector<double> > truths,
      int n_inputs);
}

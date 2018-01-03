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

%include "network.h"

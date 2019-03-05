#include <iostream>
#include "examples/Goldilocks_models/kinematics_expression.h"
#include <Eigen/Dense>

using std::cout;
using std::endl;
using Eigen::VectorXd;
using Eigen::MatrixXd;

int main() {

  int n_z = 4;
  dairlib::goldilocks_models::KinematicsExpression expr(n_z);


  // Matrix<double, Dynamic, 1> x(2);
  VectorXd x(2);
  x << 5.0 , 7.0;
  // cout << x << endl;


  auto feature = expr.getFeature(x);
  cout<< feature <<endl;




  // int dim_feature = expr.getDimFeature(x);
  // cout << "dim_feature = " << dim_feature << endl;

  // expr.getExpression();



  return 0;
}

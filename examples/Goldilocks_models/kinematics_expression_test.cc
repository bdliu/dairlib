#include <iostream>
#include "examples/Goldilocks_models/kinematics_expression.h"
#include <Eigen/Dense>

using std::cout;
using std::endl;
using Eigen::VectorXd;
using Eigen::MatrixXd;

using drake::AutoDiffVecXd;
using drake::AutoDiffXd;
using drake::math::DiscardGradient;
using drake::math::autoDiffToValueMatrix;
using drake::math::autoDiffToGradientMatrix;
using drake::math::initializeAutoDiff;

int main() {

  int n_z = 4;
  dairlib::goldilocks_models::KinematicsExpression expr(n_z);

  VectorXd x(2);
  // Matrix<double, Dynamic, 1> x(2);
  x << 1.0 , -1.0;
  AutoDiffVecXd x_autoDiff = initializeAutoDiff(x);

  ////// getFeature() //////
  VectorXd feature = expr.getFeature(x);
  cout << "feature = " << feature << endl;
  auto feature_autoDiff = expr.getFeature(x_autoDiff);
  cout << "feature_autoDiff = " << feature_autoDiff << endl;

  ////// getDimFeature() //////
  int dim_feature = expr.getDimFeature();
  cout << "dim_feature = " << dim_feature << endl;
  int dim_feature_autoDiff = expr.getDimFeature();
  cout << "dim_feature_autoDiff = " << dim_feature_autoDiff << endl;

  ///// getExpression() //////
  VectorXd theta(n_z*dim_feature);
  theta << 3.14,    1.1,
           5,       7,
           -5,      -33,
           6,       0.1;

  AutoDiffVecXd theta_autoDiff =  initializeAutoDiff(theta);
  VectorXd expression = expr.getExpression(theta,x);
  cout << "expression = " << expression << endl;
  auto expression_autoDiff = expr.getExpression(theta_autoDiff,x_autoDiff);
  cout << "expression = " << expression_autoDiff << endl;

  // Checking autoDiff



  return 0;
}

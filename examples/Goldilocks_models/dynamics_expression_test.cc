#include <iostream>
#include "math.h"
#include <Eigen/Dense>

#include "examples/Goldilocks_models/dynamics_expression.h"
#include "drake/common/drake_assert.h"

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
  int n_zDot = 6;
  int n_z = n_zDot;
  int n_feature = 5;
  dairlib::goldilocks_models::DynamicsExpression expr(n_zDot, n_feature);

  VectorXd z(n_z);
  // Matrix<double, Dynamic, 1> z(2);
  z << M_PI / 2, 3, 0,
       9, 8, 7;
  AutoDiffVecXd z_autoDiff = initializeAutoDiff(z);
  DRAKE_DEMAND(n_z == z.size());

  ////// getFeature() //////
  VectorXd feature = expr.getFeature(z);
  // cout << "feature = \n" << feature << "\n\n";
  auto feature_autoDiff = expr.getFeature(z_autoDiff);
  cout << "feature_autoDiff = \n" << feature_autoDiff << "\n\n";

  ////// getDimFeature() //////
  // int num_feature = expr.getDimFeature();
  // cout << "num_feature = \n" << num_feature << "\n\n";
  // int num_feature_autoDiff = expr.getDimFeature();
  // cout << "num_feature_autoDiff = \n" << n_feature_autoDiff << "\n\n";

  ///// getExpression() //////
  VectorXd theta = VectorXd::Zero(n_zDot/2 * n_feature);
  theta << 1, 1, 0, 0, 0,
           0, 0, 1, 0, 0,
           0, 0, 0, 1, 1;
  DRAKE_DEMAND(n_zDot/2 * n_feature == theta.size());
  // Features implemented in DynamicsExpression should be:
  // feature << z(0),
  //            z(1)*z(1)*z(1),
  //            z(0) * z(1),
  //            cos(z(0)),
  //            sqrt(z(1));

  // expression =
  //      z(0) + z(1)*z(1)*z(1),
  //      z(0) * z(1),
  //      cos(z(0)) + sqrt(z(1));

  // We don't have getExpression() that returns VectorX<double>, so we use
  // DiscardGradient here.
  VectorX<double> expression = expr.getExpression(theta, z);
  cout << "expression = \n" << expression << "\n\n";

  AutoDiffVecXd theta_autoDiff =  initializeAutoDiff(theta);
  // auto expression_autoDiff = expr.getExpression(theta_autoDiff,z_autoDiff);
  auto expression_autoDiff = expr.getExpression(theta, z_autoDiff);
  cout << "expression_autoDiff = \n" << expression_autoDiff << "\n\n";

  // Checking autoDiff
  MatrixXd jacobian  = autoDiffToGradientMatrix(expression_autoDiff);
  cout << "jacobian = \n" << jacobian << "\n\n";


  return 0;
}

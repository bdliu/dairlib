#include "examples/Goldilocks_models/kinematics_expression.h"



namespace dairlib {
namespace goldilocks_models {


KinematicsExpression::KinematicsExpression(int n_z) {
  n_feature = 2;
  n_z_ = n_z;
}

int KinematicsExpression::getDimFeature() {
  return n_feature;
}

template <typename T>
T KinematicsExpression::getExpression(T & theta, T & x) {
  // TODO(yminchen): Do I need a reference here?
  T output(n_z_);
  for (int i = 0; i < n_z_ ; i++)
    output(i) = theta.segment(i*n_feature, n_feature).dot(getFeature(x));

  return output;
}

template <typename T>
T KinematicsExpression::getFeature(T & x) {
  // TODO(yminchen): Do I need a reference here?
  T output(n_feature);

  // Implement your choice of features below
  output << x(0), x(1);
  return output;
}


// Instantiation
template VectorX<double> KinematicsExpression::getExpression(
  VectorX<double> & theta, VectorX<double> & x);
template VectorX<AutoDiffXd> KinematicsExpression::getExpression(
  VectorX<AutoDiffXd> & theta, VectorX<AutoDiffXd> & x);
template VectorX<double> KinematicsExpression::getFeature(
  VectorX<double> &);
template VectorX<AutoDiffXd> KinematicsExpression::getFeature(
  VectorX<AutoDiffXd> &);

}  // namespace goldilocks_models
}  // namespace dairlib


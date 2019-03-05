#include "examples/Goldilocks_models/kinematics_expression.h"



namespace dairlib {
namespace goldilocks_models {


KinematicsExpression::KinematicsExpression(int n_z) {
  n_feature = 5;
  n_z_ = n_z;
}

int KinematicsExpression::getDimFeature() {
  return n_feature;
}

template <typename U, typename V>
VectorX<AutoDiffXd> KinematicsExpression::getExpression(U theta, V x) {
  VectorX<AutoDiffXd> output(n_z_);
  for (int i = 0; i < n_z_ ; i++)
    output(i) = theta.segment(i*n_feature, n_feature).dot(getFeature(x));

  return output;
}

template <typename T>
T KinematicsExpression::getFeature(T x) {
  T output(n_feature);

  // Implement your choice of features below
  output << x(0),
            x(1)*x(1)*x(1),
            x(0) * x(1),
            cos(x(0)),
            sqrt(x(1));

  return output;
}


// Instantiation
// TODO(yminchen): is there a way to implement getExpression() that returns
// VectorX<double>?
template VectorX<AutoDiffXd> KinematicsExpression::getExpression(
  VectorX<double> theta, VectorX<double> x);
template VectorX<AutoDiffXd> KinematicsExpression::getExpression(
  VectorX<double> theta, VectorX<AutoDiffXd> x);
template VectorX<AutoDiffXd> KinematicsExpression::getExpression(
  VectorX<AutoDiffXd> theta, VectorX<double> x);
template VectorX<AutoDiffXd> KinematicsExpression::getExpression(
  VectorX<AutoDiffXd> theta, VectorX<AutoDiffXd> x);

template VectorX<double> KinematicsExpression::getFeature(
  VectorX<double>);
template VectorX<AutoDiffXd> KinematicsExpression::getFeature(
  VectorX<AutoDiffXd>);

}  // namespace goldilocks_models
}  // namespace dairlib


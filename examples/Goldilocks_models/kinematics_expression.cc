#include "examples/Goldilocks_models/kinematics_expression.h"


namespace dairlib {
namespace goldilocks_models {

KinematicsExpression::KinematicsExpression(int n_z, int n_feature) {
  n_feature_ = n_feature;
  n_z_ = n_z;
}

int KinematicsExpression::getDimFeature() {
  return n_feature_;
}

template <typename U, typename V>
VectorX<AutoDiffXd> KinematicsExpression::getExpression(U theta, V x) {
  VectorX<AutoDiffXd> expression(n_z_);
  for (int i = 0; i < n_z_ ; i++)
    expression(i) =
        theta.segment(i * n_feature_, n_feature_).dot(getFeature(x));

  return expression;
}

template <typename T>
T KinematicsExpression::getFeature(T x) {
  T feature(n_feature_);

  // Implement your choice of features below
  // Be careful that the dimension should match with n_feature_
  // TODO(yminchen): find a way to avoid hard coding the constraints here

  // Version 1: for kinematics_expression_test
  feature << x(0),
             x(1)*x(1)*x(1),
             x(0) * x(1),
             cos(x(0)),
             sqrt(x(1));

  return feature;
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


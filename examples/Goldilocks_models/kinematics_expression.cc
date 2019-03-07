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
VectorX<AutoDiffXd> KinematicsExpression::getExpression(
    const U & theta, const V & x) const {
  // DRAKE_DEMAND(n_z_ * n_feature_ == theta.size());  // check theta size
  // DRAKE_DEMAND(n_feature_ == getFeature(x).size());  // check feature size

  VectorX<AutoDiffXd> expression(n_z_);
  for (int i = 0; i < n_z_ ; i++)
    expression(i) =
      theta.segment(i * n_feature_, n_feature_).dot(getFeature(x));

  return expression;
}

template <typename T>
T KinematicsExpression::getFeature(const T & x) const {

  // Implement your choice of features below
  // Be careful that the dimension should match with n_feature_
  // TODO(yminchen): find a way to avoid hard coding the constraints here

  // Version 1: for kinematics_expression_test
  // T feature(5);
  // feature << x(0),
  //            x(1)*x(1)*x(1),
  //            x(0) * x(1),
  //            cos(x(0)),
  //            sqrt(x(1));

  // Version 2: testing
  T feature(1);
  feature << x(0);

  return feature;
}


// Instantiation
// TODO(yminchen): is there a way to implement getExpression() that returns
// VectorX<double>?
template VectorX<AutoDiffXd> KinematicsExpression::getExpression(
  const VectorX<double> &, const VectorX<double> &) const;
template VectorX<AutoDiffXd> KinematicsExpression::getExpression(
  const VectorX<double> &, const VectorX<AutoDiffXd> &) const;
template VectorX<AutoDiffXd> KinematicsExpression::getExpression(
  const VectorX<AutoDiffXd> &, const VectorX<double> &) const;
template VectorX<AutoDiffXd> KinematicsExpression::getExpression(
  const VectorX<AutoDiffXd> &, const VectorX<AutoDiffXd> &) const;

template VectorX<double> KinematicsExpression::getFeature(
  const VectorX<double> &) const;
template VectorX<AutoDiffXd> KinematicsExpression::getFeature(
  const VectorX<AutoDiffXd> &) const;

}  // namespace goldilocks_models
}  // namespace dairlib


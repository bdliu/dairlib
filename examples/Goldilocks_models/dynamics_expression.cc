#include "examples/Goldilocks_models/dynamics_expression.h"


namespace dairlib {
namespace goldilocks_models {

DynamicsExpression::DynamicsExpression(int n_zDot, int n_featureDot) {
  n_featureDot_ = n_featureDot;
  n_zDot_ = n_zDot;
}

int DynamicsExpression::getDimFeature() {
  return n_featureDot_;
}

template <typename U, typename V>
VectorX<AutoDiffXd> DynamicsExpression::getExpression(
    const U & theta, const V & x) const {
  // DRAKE_DEMAND(n_zDot_ * n_featureDot_ == theta.size());  // check theta size
  // DRAKE_DEMAND(n_featureDot_ == getFeature(x).size());  // check feature size

  VectorX<AutoDiffXd> expression(n_zDot_/2);
  for (int i = 0; i < n_zDot_/2 ; i++)
    expression(i) =
      theta.segment(i * n_featureDot_, n_featureDot_).dot(getFeature(x));

  return expression;
}

template <typename T>
T DynamicsExpression::getFeature(const T & x) const {

  // Implement your choice of features below
  // Be careful that the dimension should match with n_featureDot_
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
template VectorX<AutoDiffXd> DynamicsExpression::getExpression(
  const VectorX<double> &, const VectorX<double> &) const;
template VectorX<AutoDiffXd> DynamicsExpression::getExpression(
  const VectorX<double> &, const VectorX<AutoDiffXd> &) const;
template VectorX<AutoDiffXd> DynamicsExpression::getExpression(
  const VectorX<AutoDiffXd> &, const VectorX<double> &) const;
template VectorX<AutoDiffXd> DynamicsExpression::getExpression(
  const VectorX<AutoDiffXd> &, const VectorX<AutoDiffXd> &) const;

template VectorX<double> DynamicsExpression::getFeature(
  const VectorX<double> &) const;
template VectorX<AutoDiffXd> DynamicsExpression::getFeature(
  const VectorX<AutoDiffXd> &) const;

}  // namespace goldilocks_models
}  // namespace dairlib


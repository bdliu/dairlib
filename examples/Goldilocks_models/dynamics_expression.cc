#include "examples/Goldilocks_models/dynamics_expression.h"


namespace dairlib {
namespace goldilocks_models {

DynamicsExpression::DynamicsExpression(int n_sDDot, int n_feature_sDDot) {
  n_feature_sDDot_ = n_feature_sDDot;
  n_sDDot_ = n_sDDot;
}

int DynamicsExpression::getDimFeature() {
  return n_feature_sDDot_;
}

template <typename U, typename V>
V DynamicsExpression::getExpression(
    const U & theta, const V & z) const {
  // DRAKE_DEMAND(n_sDDot_ * n_feature_sDDot_ == theta.size());  // check theta size
  // DRAKE_DEMAND(n_feature_sDDot_ == getFeature(z).size());  // check feature size

  V expression(n_sDDot_);

  for (int i = 0; i < n_sDDot_; i++)
    expression(i) =
        theta.segment(i * n_feature_sDDot_, n_feature_sDDot_).dot(getFeature(z));

  return expression;
}

template <typename T>
T DynamicsExpression::getFeature(const T & z) const {

  // Implement your choice of features below
  // Be careful that the dimension should match with n_feature_sDDot_
  // TODO(yminchen): find a way to avoid hard coding the features here

  // Version 1: for dynamics_expression_test
  // T feature(5);
  // feature << z(0),
  //            z(1)*z(1)*z(1),
  //            z(0) * z(1),
  //            cos(z(0)),
  //            sqrt(z(1));

  // Version 2: testing
  T feature(1);
  feature << 0;

  // Version 3: testing
  // T feature(1);
  // feature << z(0);

  // Version 4: testing
  // T feature(2);
  // feature << z(0), z(1);

  // Version 5: nz = 4, all combinations until quadratic
  /*T feature(21);
  feature << 1,     // constant
             z(0),  // linear
             z(1),
             z(2),
             z(3),
             z(0) * z(0),  // quadratic
             z(1) * z(0),
             z(2) * z(0),
             z(3) * z(0),
             z(0) * z(1),
             z(1) * z(1),
             z(2) * z(1),
             z(3) * z(1),
             z(0) * z(2),
             z(1) * z(2),
             z(2) * z(2),
             z(3) * z(2),
             z(0) * z(3),
             z(1) * z(3),
             z(2) * z(3),
             z(3) * z(3);*/


  return feature;
}


// Instantiation
// TODO(yminchen): is there a way to implement getExpression() that returns
// VectorX<double>?
template VectorX<double> DynamicsExpression::getExpression(
  const VectorX<double> &, const VectorX<double> &) const;
template VectorX<AutoDiffXd> DynamicsExpression::getExpression(
  const VectorX<double> &, const VectorX<AutoDiffXd> &) const;
// template VectorX<AutoDiffXd> DynamicsExpression::getExpression(
//   const VectorX<AutoDiffXd> &, const VectorX<double> &) const;
// template VectorX<AutoDiffXd> DynamicsExpression::getExpression(
//   const VectorX<AutoDiffXd> &, const VectorX<AutoDiffXd> &) const;

template VectorX<double> DynamicsExpression::getFeature(
  const VectorX<double> &) const;
template VectorX<AutoDiffXd> DynamicsExpression::getFeature(
  const VectorX<AutoDiffXd> &) const;

}  // namespace goldilocks_models
}  // namespace dairlib


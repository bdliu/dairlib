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
    const U & theta, const V & s, const V & ds) const {
  // DRAKE_DEMAND(n_sDDot_ * n_feature_sDDot_ == theta.size());  // check theta size
  // DRAKE_DEMAND(n_feature_sDDot_ == getFeature(s).size());  // check feature size

  V expression(n_sDDot_);

  for (int i = 0; i < n_sDDot_; i++)
    expression(i) =
        theta.segment(i * n_feature_sDDot_, n_feature_sDDot_).dot(getFeature(s,ds));

  return expression;
}

template <typename T>
T DynamicsExpression::getFeature(const T & s, const T & ds) const {

  // Implement your choice of features below
  // Be careful that the dimension should match with n_feature_sDDot_
  // TODO(yminchen): find a way to avoid hard coding the features here

  // Version 1: for dynamics_expression_test
  // T feature(5);
  // feature << s(0),
  //            s(1)*s(1)*s(1),
  //            s(0) * s(1),
  //            cos(s(0)),
  //            sqrt(s(1));

  // Version 2: testing
  // T feature(1);
  // feature << 0;

  // Version 3: testing
  // T feature(1);
  // feature << s(0);

  // Version 4: testing
  // T feature(2);
  // feature << s(0), s(1);

  // Version 5: ns = 2, all combinations until quadratic
  // DRAKE_DEMAND(n_sDDot_ == 2);
  // T feature(21);
  // feature << 1,     // constant
  //            s(0),  // linear
  //            s(1),
  //            ds(0),
  //            ds(1),
  //            s(0) * s(0),  // quadratic
  //            s(1) * s(0),
  //            ds(0) * s(0),
  //            ds(1) * s(0),
  //            s(0) * s(1),
  //            s(1) * s(1),
  //            ds(0) * s(1),
  //            ds(1) * s(1),
  //            s(0) * ds(0),
  //            s(1) * ds(0),
  //            ds(0) * ds(0),
  //            ds(1) * ds(0),
  //            s(0) * ds(1),
  //            s(1) * ds(1),
  //            ds(0) * ds(1),
  //            ds(1) * ds(1);

  // Version 5: ns = 1, all combinations until quadratic
  /*DRAKE_DEMAND(n_sDDot_ == 1);
  T feature(6);
  feature << 1,     // constant
             s(0),  // linear
             ds(0),
             s(0) * s(0),  // quadratic
             ds(0) * s(0),
             ds(0) * ds(0);*/

  // Version 6: testing
  DRAKE_DEMAND(n_sDDot_ == 1);
  T feature(1);
  feature << s(0)*s(0)*s(0);

  return feature;
}


// Instantiation
// TODO(yminchen): is there a way to implement getExpression() that returns
// VectorX<double>?
template VectorX<double> DynamicsExpression::getExpression(
  const VectorX<double> &, const VectorX<double> &, const VectorX<double> &) const;
template VectorX<AutoDiffXd> DynamicsExpression::getExpression(
  const VectorX<double> &, const VectorX<AutoDiffXd> &, const VectorX<AutoDiffXd> &) const;
// template VectorX<AutoDiffXd> DynamicsExpression::getExpression(
//   const VectorX<AutoDiffXd> &, const VectorX<double> &, const VectorX<double> &) const;
// template VectorX<AutoDiffXd> DynamicsExpression::getExpression(
//   const VectorX<AutoDiffXd> &, const VectorX<AutoDiffXd> &, const VectorX<AutoDiffXd> &) const;

template VectorX<double> DynamicsExpression::getFeature(
  const VectorX<double> &, const VectorX<double> &) const;
template VectorX<AutoDiffXd> DynamicsExpression::getFeature(
  const VectorX<AutoDiffXd> &, const VectorX<AutoDiffXd> &) const;

}  // namespace goldilocks_models
}  // namespace dairlib


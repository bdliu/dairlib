#include "examples/Goldilocks_models/kinematics_expression.h"



namespace dairlib {
namespace goldilocks_models {


KinematicsExpression::KinematicsExpression(int n_z) {
  n_feature_ = 1;
  n_z_ = n_z;
}

template <typename T>
MatrixX<T> & KinematicsExpression::getExpression(
    MatrixX<T> & theta, MatrixX<T>& x) {

  // implement theta*getFeature(x)
  return theta.reshaped(n_z_, n_feature_) * getFeature(x);
}

template <typename T>
T KinematicsExpression::getFeature(
    T & x) {
  // TODO(yminchen): Do I need a reference here?
  T output(n_feature_);

  // Implement your choice of features below
  output << x(0);
  return output;
}


template <typename T>
int KinematicsExpression::getDimFeature(MatrixX<T> & x) {
  return getFeature(x).size();
}



// Instantiation
// template Matrix<double, Dynamic, 1> & KinematicsExpression::getExpression(
//     Matrix<double, Dynamic, 1> & theta, Matrix<double, Dynamic, 1>& x);
// template Matrix<AutoDiffXd, Dynamic, 1> & KinematicsExpression::getExpression(
//     Matrix<AutoDiffXd, Dynamic, 1> & theta, Matrix<AutoDiffXd, Dynamic, 1>& x);
template VectorX<double> KinematicsExpression::getFeature(
    VectorX<double> &);
template VectorX<AutoDiffXd> KinematicsExpression::getFeature(
    VectorX<AutoDiffXd> & );
// template Matrix<AutoDiffXd, Dynamic, 1> KinematicsExpression::getFeature(
//     Matrix<AutoDiffXd, Dynamic, 1>);
// template int KinematicsExpression::getDimFeature(
//     Matrix<double, Dynamic, 1>);
// template int KinematicsExpression::getDimFeature(
//     Matrix<AutoDiffXd, Dynamic, 1>);



}  // namespace goldilocks_models
}  // namespace dairlib


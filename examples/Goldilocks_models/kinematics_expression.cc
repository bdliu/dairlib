#include "examples/Goldilocks_models/kinematics_expression.h"



namespace dairlib {
namespace goldilocks_models {


template <typename T>
KinematicsExpression<T>::KinematicsExpression(int n_z) {
  n_feature_ = 1;
  n_z_ = n_z;
}

template <typename T>
Matrix<T, Dynamic, 1> & KinematicsExpression<T>::getExpression(
  Matrix<T, Dynamic, 1> & theta, Matrix<T, Dynamic, 1>& x) {

  // implement theta*getFeature(x)
  return theta.reshaped(n_z_, n_feature_) * getFeature(x);
}

template <typename T>
Matrix<T, Dynamic, 1> & KinematicsExpression<T>::getFeature(
  Matrix<T, Dynamic, 1> & x) {
  // TODO(yminchen): Do I need a reference here?
  Matrix<T, Dynamic, 1> output(n_feature_);

  // Implement your choice of features below
  output << x(0);
  return output;
}

template <typename T>
int KinematicsExpression<T>::getDimFeature(Matrix<T, Dynamic, 1> & x) {
  return getFeature(x).size();
}


}  // namespace goldilocks_models
}  // namespace dairlib


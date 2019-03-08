#include "examples/Goldilocks_models/kinematics_expression.h"


namespace dairlib {
namespace goldilocks_models {

template <typename T>
KinematicsExpression<T>::KinematicsExpression(int n_z, int n_feature) {
  n_feature_ = n_feature;
  n_z_ = n_z;
}

template <typename T>
KinematicsExpression<T>::KinematicsExpression(int n_z, int n_feature,
    const MultibodyPlant<T> * plant) {
  n_feature_ = n_feature;
  n_z_ = n_z;
  plant_ = plant;
}

template <typename T>
int KinematicsExpression<T>::getDimFeature() {
  return n_feature_;
}

template <typename T>
template <typename U, typename V>
VectorX<T> KinematicsExpression<T>::getExpression(
    const U & theta, const V & x) const {
  // DRAKE_DEMAND(n_z_ * n_feature_ == theta.size());  // check theta size
  // DRAKE_DEMAND(n_feature_ == getFeature(x).size());  // check feature size

  VectorX<T> expression(n_z_);
  for (int i = 0; i < n_z_ ; i++)
    expression(i) =
      theta.segment(i * n_feature_, n_feature_).dot(getFeature(x));

  return expression;
}

template <typename T>
template <typename U>
VectorX<U> KinematicsExpression<T>::getFeature(const VectorX<U> & x) const {

  // Implement your choice of features below
  // Be careful that the dimension should match with n_feature_
  // TODO(yminchen): find a way to avoid hard coding the features here

  //////////// Version 1: for kinematics_expression_test ///////////////////////
  VectorX<U> feature(5);
  feature << x(0),
             x(1)*x(1)*x(1),
             x(0) * x(1),
             cos(x(0)),
             sqrt(x(1));

  //////////// Version 2: testing //////////////////////////////////////////////
  // VectorX<U> feature(1);
  // feature << x(0);

  //////////// Version 3: SLIP /////////////////////////////////////////////////
  // Get CoM position and stance foot position in autoDiff
  auto context = plant_->CreateDefaultContext();
  plant_->SetPositionsAndVelocities(context.get(), x);

  // cout << "num_model_instances() = " << plant_->num_model_instances() << endl;
  // cout << "x = " << x << endl;

  // const Body< T > &   GetBodyByName (const std::string &name)
  // const std::string &   GetModelInstanceName (ModelInstanceIndex model_instance) const

  return feature;
}


// Instantiation

// class KinematicsExpression
template class KinematicsExpression<double>;
template class KinematicsExpression<AutoDiffXd>;

// method getExpression
template VectorX<double> KinematicsExpression<double>::getExpression(
  const VectorX<double> &, const VectorX<double> &) const;

template VectorX<AutoDiffXd> KinematicsExpression<AutoDiffXd>::getExpression(
  const VectorX<double> &, const VectorX<double> &) const;
template VectorX<AutoDiffXd> KinematicsExpression<AutoDiffXd>::getExpression(
  const VectorX<double> &, const VectorX<AutoDiffXd> &) const;
template VectorX<AutoDiffXd> KinematicsExpression<AutoDiffXd>::getExpression(
  const VectorX<AutoDiffXd> &, const VectorX<double> &) const;
template VectorX<AutoDiffXd> KinematicsExpression<AutoDiffXd>::getExpression(
  const VectorX<AutoDiffXd> &, const VectorX<AutoDiffXd> &) const;

// method getFeature
template VectorX<double> KinematicsExpression<double>::getFeature(
  const VectorX<double> &) const;

template VectorX<double> KinematicsExpression<AutoDiffXd>::getFeature(
  const VectorX<double> &) const;
template VectorX<AutoDiffXd> KinematicsExpression<AutoDiffXd>::getFeature(
  const VectorX<AutoDiffXd> &) const;

}  // namespace goldilocks_models
}  // namespace dairlib


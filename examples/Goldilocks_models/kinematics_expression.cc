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
  // VectorX<U> feature(5);
  // feature << x(0),
  //            x(1)*x(1)*x(1),
  //            x(0) * x(1),
  //            cos(x(0)),
  //            sqrt(x(1));

  //////////// Version 2: testing //////////////////////////////////////////////
  VectorX<U> feature(1);
  feature << x(0);

  //////////// Version 3: SLIP /////////////////////////////////////////////////
  /*// If you use plant functions, then it's required that T = U?
  // Get CoM position and stance foot position in autoDiff
  auto context = plant_->CreateDefaultContext();
  plant_->SetPositionsAndVelocities(context.get(), x);

  // const Body< T > &   GetBodyByName (const std::string &name)
  const auto & torso = plant_->GetBodyByName("torso_mass");
  const auto & left_upper_leg = plant_->GetBodyByName("left_upper_leg_mass");
  const auto & left_lower_leg = plant_->GetBodyByName("left_lower_leg_mass");
  const auto & right_upper_leg = plant_->GetBodyByName("right_upper_leg_mass");
  const auto & right_lower_leg = plant_->GetBodyByName("right_lower_leg_mass");

  // const Isometry3< T > &  EvalBodyPoseInWorld (
  //  const systems::Context< T > &context, const Body< T > &body_B)
  const auto & torso_pose = plant_->EvalBodyPoseInWorld(
                              *context, torso);
  const auto & left_upper_leg_pose = plant_->EvalBodyPoseInWorld(
                                       *context, left_upper_leg);
  const auto & left_lower_leg_pose = plant_->EvalBodyPoseInWorld(
                                       *context, left_lower_leg);
  const auto & right_upper_leg_pose = plant_->EvalBodyPoseInWorld(
                                        *context, right_upper_leg);
  const auto & right_lower_leg_pose = plant_->EvalBodyPoseInWorld(
                                        *context, right_lower_leg);
  */


  return feature;
}


// Instantiation

// class KinematicsExpression
template class KinematicsExpression<double>;
template class KinematicsExpression<AutoDiffXd>;

// method getExpression
template VectorX<double> KinematicsExpression<double>::getExpression(
  const VectorX<double> &, const VectorX<double> &) const;

// template VectorX<AutoDiffXd> KinematicsExpression<AutoDiffXd>::getExpression(
//   const VectorX<double> &, const VectorX<double> &) const;
template VectorX<AutoDiffXd> KinematicsExpression<AutoDiffXd>::getExpression(
  const VectorX<double> &, const VectorX<AutoDiffXd> &) const;
// template VectorX<AutoDiffXd> KinematicsExpression<AutoDiffXd>::getExpression(
//   const VectorX<AutoDiffXd> &, const VectorX<double> &) const;
template VectorX<AutoDiffXd> KinematicsExpression<AutoDiffXd>::getExpression(
  const VectorX<AutoDiffXd> &, const VectorX<AutoDiffXd> &) const;

// method getFeature
template VectorX<double> KinematicsExpression<double>::getFeature(
  const VectorX<double> &) const;

// template VectorX<double> KinematicsExpression<AutoDiffXd>::getFeature(
//   const VectorX<double> &) const;
template VectorX<AutoDiffXd> KinematicsExpression<AutoDiffXd>::getFeature(
  const VectorX<AutoDiffXd> &) const;

}  // namespace goldilocks_models
}  // namespace dairlib


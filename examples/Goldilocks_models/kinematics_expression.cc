#include "examples/Goldilocks_models/kinematics_expression.h"


namespace dairlib {
namespace goldilocks_models {

template <typename T>
KinematicsExpression<T>::KinematicsExpression(int n_s, int n_feature) {
  n_feature_ = n_feature;
  n_s_ = n_s;
}

template <typename T>
KinematicsExpression<T>::KinematicsExpression(int n_s, int n_feature,
    const MultibodyPlant<T> * plant) {
  n_feature_ = n_feature;
  n_s_ = n_s;
  plant_ = plant;

  mass_disp_ << 0, 0, -0.25;
  foot_disp_ << 0, 0, -0.5;
}

template <typename T>
int KinematicsExpression<T>::getDimFeature() {
  return n_feature_;
}

template <typename T>
template <typename U, typename V>
VectorX<T> KinematicsExpression<T>::getExpression(
  const U & theta, const V & q) const {
  // DRAKE_DEMAND(n_s_ * n_feature_ == theta.size());  // check theta size
  // DRAKE_DEMAND(n_feature_ == getFeature(q).size());  // check feature size

  VectorX<T> expression(n_s_);
  for (int i = 0; i < n_s_ ; i++)
    expression(i) =
      theta.segment(i * n_feature_, n_feature_).dot(getFeature(q));

  return expression;
}

template <typename T>
template <typename U>
VectorX<U> KinematicsExpression<T>::getFeature(const VectorX<U> & q) const {

  // Implement your choice of features below
  // Be careful that the dimension should match with n_feature_
  // TODO(yminchen): find a way to avoid hard coding the features here

  //////////// Version 1: for kinematics_expression_test ///////////////////////
  // VectorX<U> feature(5);
  // feature << q(0),
  //            q(1)*q(1)*q(1),
  //            q(0) * q(1),
  //            cos(q(0)),
  //            sqrt(q(1));

  //////////// Version 2: testing //////////////////////////////////////////////
  // VectorX<U> feature(1);
  // feature << q(1);

  //////////// Version 3: testing //////////////////////////////////////////////
  // VectorX<U> feature(2);
  // feature << q(1), q(1+q.size()/2);

  //////////// Version 4: testing MBP //////////////////////////////////////////
  // // If you use plant functions, then it's required that T = U?
  // auto context = plant_->CreateDefaultContext();
  // plant_->SetPositions(context.get(), q);

  // const auto & right_lower_leg = plant_->GetBodyByName("right_lower_leg_mass");
  // const auto & right_lower_leg_pose = plant_->EvalBodyPoseInWorld(
  //                                       *context, right_lower_leg);

  // Vector3d disp(0, 0, -0.5);
  // const VectorX<U> right_lower_leg_Pos = right_lower_leg_pose.translation();
  // const MatrixX<U> right_lower_leg_Rotmat = right_lower_leg_pose.linear();
  // VectorX<U> feature = right_lower_leg_Pos + right_lower_leg_Rotmat * disp;

  //////////// Version 5: testing r_stance_foot_to_CoM /////////////////////////
  /*// If you use plant functions, then it's required that T = U?
  // Get CoM position and stance foot position in autoDiff
  auto context = plant_->CreateDefaultContext();
  plant_->SetPositions(context.get(), q);

  // CoM
  const auto & torso = plant_->GetBodyByName("torso_mass");
  const auto & torso_pose = plant_->EvalBodyPoseInWorld(*context, torso);
  VectorX<U> CoM = 0.5 * torso_pose.translation();

  for (int i = 0; i < 4; i++) {
    const auto & body = plant_->GetBodyByName(leg_link_names_[i]);
    const auto & body_pose = plant_->EvalBodyPoseInWorld(*context, body);

    CoM += (body_pose.translation() + body_pose.linear() * mass_disp_) / 8.0;
  }

  // Stance foot position (left foot)
  const auto & left_lower_leg = plant_->GetBodyByName("left_lower_leg_mass");
  const auto & left_lower_leg_pose = plant_->EvalBodyPoseInWorld(
                                        *context, left_lower_leg);
  const VectorX<U> left_lower_leg_Pos = left_lower_leg_pose.translation();
  const MatrixX<U> left_lower_leg_Rotmat = left_lower_leg_pose.linear();
  VectorX<U> left_foot_pos = left_lower_leg_Pos + left_lower_leg_Rotmat * foot_disp_;

  VectorX<U> foot_to_CoM = CoM - left_foot_pos;
  VectorX<U> feature(1);
  feature << foot_to_CoM.norm();*/

  //////////// Version 6: SLIP /////////////////////////////////////////////////
/*  // If you use plant functions, then it's required that T = U?
  // Get CoM position and stance foot position in autoDiff
  auto context = plant_->CreateDefaultContext();
  plant_->SetPositions(context.get(), q);

  // CoM
  const auto & torso = plant_->GetBodyByName("torso_mass");
  const auto & torso_pose = plant_->EvalBodyPoseInWorld(*context, torso);
  VectorX<U> CoM = 0.5 * torso_pose.translation();

  for (int i = 0; i < 4; i++) {
    const auto & body = plant_->GetBodyByName(leg_link_names_[i]);
    const auto & body_pose = plant_->EvalBodyPoseInWorld(*context, body);

    CoM += (body_pose.translation() + body_pose.linear() * mass_disp_) / 8.0;
  }

  // Stance foot position (left foot)
  const auto & left_lower_leg = plant_->GetBodyByName("left_lower_leg_mass");
  const auto & left_lower_leg_pose = plant_->EvalBodyPoseInWorld(
                                        *context, left_lower_leg);
  const VectorX<U> left_lower_leg_Pos = left_lower_leg_pose.translation();
  const MatrixX<U> left_lower_leg_Rotmat = left_lower_leg_pose.linear();
  VectorX<U> left_foot_pos = left_lower_leg_Pos + left_lower_leg_Rotmat * foot_disp_;

  VectorX<U> foot_to_CoM = CoM - left_foot_pos;

  VectorX<U> feature(2);
  feature << foot_to_CoM.norm(),
             tan(foot_to_CoM(0)/foot_to_CoM(2));*/

  //////////// Version 7: Quadratic combinations /////////////////////////////
  // elements:
  // q(0),
  // q(1),
  // sin(q(2)), cos(q(2))
  // sin(q(3)), cos(q(3))
  // sin(q(4)), cos(q(4))
  // sin(q(5)), cos(q(5))
  // sin(q(6)), cos(q(6))

  /*VectorX<U> feature(68); //1 + 12 + 10C2 = 1 + 12 + 55 = 68
  feature <<1,
            q(0),
            q(1),
            sin(q(2)),
            cos(q(2)),
            sin(q(3)),
            cos(q(3)),
            sin(q(4)),
            cos(q(4)),
            sin(q(5)),
            cos(q(5)),
            sin(q(6)),
            cos(q(6)), // linear until here, below are quadratic
            // 1
            sin(q(2)) * sin(q(2)),
            cos(q(2)) * sin(q(2)),
            sin(q(3)) * sin(q(2)),
            cos(q(3)) * sin(q(2)),
            sin(q(4)) * sin(q(2)),
            cos(q(4)) * sin(q(2)),
            sin(q(5)) * sin(q(2)),
            cos(q(5)) * sin(q(2)),
            sin(q(6)) * sin(q(2)),
            cos(q(6)) * sin(q(2)),
            // 2
            cos(q(2)) * cos(q(2)),
            sin(q(3)) * cos(q(2)),
            cos(q(3)) * cos(q(2)),
            sin(q(4)) * cos(q(2)),
            cos(q(4)) * cos(q(2)),
            sin(q(5)) * cos(q(2)),
            cos(q(5)) * cos(q(2)),
            sin(q(6)) * cos(q(2)),
            cos(q(6)) * cos(q(2)),
            // 3
            sin(q(3)) * sin(q(3)),
            cos(q(3)) * sin(q(3)),
            sin(q(4)) * sin(q(3)),
            cos(q(4)) * sin(q(3)),
            sin(q(5)) * sin(q(3)),
            cos(q(5)) * sin(q(3)),
            sin(q(6)) * sin(q(3)),
            cos(q(6)) * sin(q(3)),
            // 4
            cos(q(3)) * cos(q(3)),
            sin(q(4)) * cos(q(3)),
            cos(q(4)) * cos(q(3)),
            sin(q(5)) * cos(q(3)),
            cos(q(5)) * cos(q(3)),
            sin(q(6)) * cos(q(3)),
            cos(q(6)) * cos(q(3)),
            // 5
            sin(q(4)) * sin(q(4)),
            cos(q(4)) * sin(q(4)),
            sin(q(5)) * sin(q(4)),
            cos(q(5)) * sin(q(4)),
            sin(q(6)) * sin(q(4)),
            cos(q(6)) * sin(q(4)),
            // 6
            cos(q(4)) * cos(q(4)),
            sin(q(5)) * cos(q(4)),
            cos(q(5)) * cos(q(4)),
            sin(q(6)) * cos(q(4)),
            cos(q(6)) * cos(q(4)),
            // 7
            sin(q(5)) * sin(q(5)),
            cos(q(5)) * sin(q(5)),
            sin(q(6)) * sin(q(5)),
            cos(q(6)) * sin(q(5)),
            // 8
            cos(q(5)) * cos(q(5)),
            sin(q(6)) * cos(q(5)),
            cos(q(6)) * cos(q(5)),
            // 9
            sin(q(6)) * sin(q(6)),
            cos(q(6)) * sin(q(6)),
            // 10
            cos(q(6)) * cos(q(6));*/

  //////////// Version 8: debug B matrix ///////////////////////////////////////
  VectorX<U> feature(1);
  feature << q(1);

  return feature;
}



// Instantiation

// class KinematicsExpression //////////////////////////////////////////////////
template class KinematicsExpression<double>;
template class KinematicsExpression<AutoDiffXd>;

// method getExpression ////////////////////////////////////////////////////////
template VectorX<double> KinematicsExpression<double>::getExpression(
  const VectorX<double> &, const VectorX<double> &) const;

// template VectorX<AutoDiffXd> KinematicsExpression<AutoDiffXd>::getExpression(
//   const VectorX<double> &, const VectorX<double> &) const;
template VectorX<AutoDiffXd> KinematicsExpression<AutoDiffXd>::getExpression(
  const VectorX<double> &, const VectorX<AutoDiffXd> &) const;
template VectorX<AutoDiffXd> KinematicsExpression<AutoDiffXd>::getExpression(
  const VectorX<AutoDiffXd> &, const VectorX<double> &) const;

// Seems that the case when theta and q are both autoDiff doesn't work. (I
// tested with an example.)
// template VectorX<AutoDiffXd> KinematicsExpression<AutoDiffXd>::getExpression(
//   const VectorX<AutoDiffXd> &, const VectorX<AutoDiffXd> &) const;

// method getFeature ///////////////////////////////////////////////////////////
template VectorX<double> KinematicsExpression<double>::getFeature(
  const VectorX<double> &) const;

// template VectorX<double> KinematicsExpression<AutoDiffXd>::getFeature(
//   const VectorX<double> &) const;
template VectorX<AutoDiffXd> KinematicsExpression<AutoDiffXd>::getFeature(
  const VectorX<AutoDiffXd> &) const;

}  // namespace goldilocks_models
}  // namespace dairlib


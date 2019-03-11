#include "examples/Goldilocks_models/kinematics_constraint.h"


namespace dairlib {
namespace goldilocks_models {

KinematicsConstraint::KinematicsConstraint(
                                 int n_z, int n_feature, VectorXd & thetaZ,
                                 const MultibodyPlant<AutoDiffXd> * plant,
                                 const std::string& description):
  Constraint(n_z,
             n_z + plant->num_positions() + plant->num_velocities(),
             VectorXd::Zero(n_z),
             VectorXd::Zero(n_z),
             description),
  plant_(plant),
  n_constraint_(n_z),
  n_feature_(n_feature),
  thetaZ_(thetaZ),
  n_state_(plant->num_positions() + plant->num_velocities()),
  expression_double(KinematicsExpression<double>(n_z, n_feature)),
  expression_autoDiff_(KinematicsExpression<AutoDiffXd>(n_z, n_feature)) {

  // Check the theta size
  DRAKE_DEMAND(n_z * n_feature == thetaZ.size());

  // Check the feature size implemented in the model expression
  VectorXd x_temp = VectorXd::Zero(
      plant->num_positions() + plant->num_velocities());
  DRAKE_DEMAND(n_feature == expression_double.getFeature(x_temp).size());
}

void KinematicsConstraint::DoEval(const
                             Eigen::Ref<const Eigen::VectorXd>& z_x,
                             Eigen::VectorXd* y) const {
  AutoDiffVecXd y_t;
  Eval(initializeAutoDiff(z_x), &y_t);
  *y = autoDiffToValueMatrix(y_t);
}

void KinematicsConstraint::DoEval(const
                             Eigen::Ref<const AutoDiffVecXd>& z_x,
                             AutoDiffVecXd* y) const {
  const AutoDiffVecXd z = z_x.head(n_constraint_);
  const AutoDiffVecXd x = z_x.tail(n_state_);

  *y = getKinematicsConstraint(z, x, thetaZ_);
}

void KinematicsConstraint::DoEval(const
                             Eigen::Ref<const VectorX<Variable>>& x,
                             VectorX<Expression>*y) const {
  throw std::logic_error(
    "This constraint class does not support symbolic evaluation.");
}

VectorXd KinematicsConstraint::getGradientWrtTheta(VectorXd & x){
  VectorXd z = VectorXd::Zero(n_constraint_);
  VectorXd gradient(n_feature_);
  for(int i = 0; i<n_feature_; i++){
    VectorXd theta_unit = VectorXd::Zero(thetaZ_.size());
    theta_unit(i) = 1;
    gradient(i) = getKinematicsConstraint(z,x,theta_unit)(0);
  }
  return gradient;
}

AutoDiffVecXd KinematicsConstraint::getKinematicsConstraint(
  const AutoDiffVecXd & z, const AutoDiffVecXd & x, const VectorXd & theta) const{
  return z - expression_autoDiff_.getExpression(theta, x);
}
VectorXd KinematicsConstraint::getKinematicsConstraint(
  const VectorXd & z, const VectorXd & x, const VectorXd & theta) const{
  return z - expression_double.getExpression(theta, x);
}




}  // namespace goldilocks_models
}  // namespace dairlib

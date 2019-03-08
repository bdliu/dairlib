#include "examples/Goldilocks_models/kinematics_constraint.h"


namespace dairlib {
namespace goldilocks_models {

KinematicsConstraint::KinematicsConstraint(
                                 int n_z, int n_feature, int n_theta,
                                 const MultibodyPlant<double> * plant,
                                 const std::string& description):
  Constraint(n_z,
             n_z + n_theta + plant->num_positions() + plant->num_velocities(),
             VectorXd::Zero(n_z),
             VectorXd::Zero(n_z),
             description),
  plant_(plant),
  n_constraint_(n_z),
  n_feature_(n_feature),
  n_theta_(n_theta),
  expression_object_(KinematicsExpression<AutoDiffXd>(n_z, n_feature)) {

  // Check the theta size
  DRAKE_DEMAND(n_z * n_feature == n_theta);

  // Check the feature size implemented in the model expression
  VectorXd x_temp = VectorXd::Zero(
      plant->num_positions() + plant->num_velocities());
  DRAKE_DEMAND(n_feature == expression_object_.getFeature(x_temp).size());
}

void KinematicsConstraint::DoEval(const
                             Eigen::Ref<const Eigen::VectorXd>& z_theta_x,
                             Eigen::VectorXd* y) const {
  AutoDiffVecXd y_t;
  Eval(initializeAutoDiff(z_theta_x), &y_t);
  *y = autoDiffToValueMatrix(y_t);
}

void KinematicsConstraint::DoEval(const
                             Eigen::Ref<const AutoDiffVecXd>& z_theta_x,
                             AutoDiffVecXd* y) const {
  const AutoDiffVecXd z = z_theta_x.head(n_constraint_);
  const AutoDiffVecXd theta = z_theta_x.segment(n_constraint_, n_theta_);
  const AutoDiffVecXd x = z_theta_x.tail(
      plant_->num_positions() + plant_->num_velocities());

  *y = z - expression_object_.getExpression(theta, x);
}

void KinematicsConstraint::DoEval(const
                             Eigen::Ref<const VectorX<Variable>>& x,
                             VectorX<Expression>*y) const {
  throw std::logic_error(
    "This constraint class does not support symbolic evaluation.");
}




}  // namespace goldilocks_models
}  // namespace dairlib

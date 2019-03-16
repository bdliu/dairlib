#include "examples/Goldilocks_models/dynamics_constraint.h"


namespace dairlib {
namespace goldilocks_models {

DynamicsConstraint::DynamicsConstraint(
  int n_zDDot, int n_featureDot,
  VectorXd & thetaZDDot,
  const MultibodyPlant<AutoDiffXd> * plant,
  const std::string& description):
  Constraint(n_zDDot,
             2 * n_zDDot + 1,
             VectorXd::Zero(n_zDDot),
             VectorXd::Zero(n_zDDot),
             description),
  plant_(plant),
  n_zDDot_(n_zDDot),
  n_featureZDDot_(n_featureDot),
  thetaZDDot_(thetaZDDot),
  expression_object_(DynamicsExpression(n_zDDot, n_featureDot)) {

  // Check the theta size
  DRAKE_DEMAND(n_zDDot * n_featureDot == thetaZDDot.size());

  // Check the feature size implemented in the model expression
  VectorXd z_temp = VectorXd::Zero(n_zDDot);
  DRAKE_DEMAND(n_featureDot == expression_object_.getFeature(z_temp).size());
}


void DynamicsConstraint::DoEval(const
                                Eigen::Ref<const Eigen::VectorXd>& q,
                                Eigen::VectorXd* y) const {
  AutoDiffVecXd y_t;
  Eval(initializeAutoDiff(q), &y_t);
  *y = autoDiffToValueMatrix(y_t);
}

void DynamicsConstraint::DoEval(const
                                Eigen::Ref<const AutoDiffVecXd>& q,
                                AutoDiffVecXd* y) const {
  const AutoDiffVecXd z_i = q.head(n_zDDot_);
  const AutoDiffVecXd z_iplus1 = q.segment(n_zDDot_, n_zDDot_);
  const AutoDiffVecXd timestep_i = q.tail(1);

  *y = getDynamicsConstraint(z_i, z_iplus1, timestep_i, thetaZDDot_);
}

void DynamicsConstraint::DoEval(const
                                Eigen::Ref<const VectorX<Variable>>& x,
                                VectorX<Expression>*y) const {
  throw std::logic_error(
    "This constraint class does not support symbolic evaluation.");
}


VectorXd DynamicsConstraint::getGradientWrtTheta(
  const VectorXd & z_i, const VectorXd & z_iplus1,
  const VectorXd & timestep_i) const {
  VectorXd gradient(n_featureZDDot_);
  for (int i = 0; i < n_featureZDDot_; i++) {
    VectorXd theta_unit = VectorXd::Zero(thetaZDDot_.size());
    theta_unit(i) = 1;
    gradient(i) = getDynamicsConstraint(z_i, z_iplus1, timestep_i,
                                        theta_unit)(0) - z_iplus1(0) + z_i(0);
  }
  // TODO(yminchen): You can also use autoDiff to get the gradient herre.
  return gradient;
}



AutoDiffVecXd DynamicsConstraint::getDynamicsConstraint(
  const AutoDiffVecXd & z_i, const AutoDiffVecXd & z_iplus1,
  const AutoDiffVecXd & timestep_i, const VectorXd & theta) const {

  // Let the dynamics be dzdt = h(z;thetaZDDot) = h(z).
  AutoDiffVecXd h_of_z_i = expression_object_.getExpression(theta,
                           z_i);
  AutoDiffVecXd h_of_z_iplus1 = expression_object_.getExpression(theta,
                                z_iplus1);

  // Collocation point
  AutoDiffVecXd z_collocation = (z_i + z_iplus1) / 2 +
                                (h_of_z_i - h_of_z_iplus1) * timestep_i(0) / 8;

  AutoDiffVecXd h_of_colloc_pt = expression_object_.getExpression(theta,
                                 z_collocation);

  return (z_iplus1 - z_i) - timestep_i(0) * (h_of_z_i + 4 * h_of_colloc_pt +
         h_of_z_iplus1) / 6;
}
VectorXd DynamicsConstraint::getDynamicsConstraint(
  const VectorXd & z_i, const VectorXd & z_iplus1,
  const VectorXd & timestep_i, const VectorXd & theta) const {

  // Let the dynamics be dzdt = h(z;thetaZDDot) = h(z).
  VectorXd h_of_z_i =  expression_object_.getExpression(theta,
                       z_i);
  VectorXd h_of_z_iplus1 = expression_object_.getExpression(theta,
                           z_iplus1);

  // Collocation point
  VectorXd z_collocation = (z_i + z_iplus1) / 2 +
                           (h_of_z_i - h_of_z_iplus1) * timestep_i(0) / 8;

  VectorXd h_of_colloc_pt = expression_object_.getExpression(theta,
                            z_collocation);

  return (z_iplus1 - z_i) - timestep_i(0) * (h_of_z_i + 4 * h_of_colloc_pt +
         h_of_z_iplus1) / 6;
}



}  // namespace goldilocks_models
}  // namespace dairlib

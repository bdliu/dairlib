#include "examples/Goldilocks_models/dynamics_constraint.h"


namespace dairlib {
namespace goldilocks_models {

DynamicsConstraint::DynamicsConstraint(
  int n_sDDot, int n_feature_sDDot,
  VectorXd & theta_sDDot,
  const MultibodyPlant<AutoDiffXd> * plant,
  const std::string& description):
  Constraint(n_sDDot,
             2 * n_sDDot + 1,
             VectorXd::Zero(n_sDDot),
             VectorXd::Zero(n_sDDot),
             description),
  plant_(plant),
  n_sDDot_(n_sDDot),
  n_feature_sDDot_(n_feature_sDDot),
  theta_sDDot_(theta_sDDot),
  expression_object_(DynamicsExpression(n_sDDot, n_feature_sDDot)) {

  // Check the theta size
  DRAKE_DEMAND(n_sDDot * n_feature_sDDot == theta_sDDot.size());

  // Check the feature size implemented in the model expression
  VectorXd z_temp = VectorXd::Zero(n_sDDot);
  DRAKE_DEMAND(n_feature_sDDot == expression_object_.getFeature(z_temp).size());
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
  const AutoDiffVecXd s_i = q.head(n_sDDot_);
  const AutoDiffVecXd s_iplus1 = q.segment(n_sDDot_, n_sDDot_);
  const AutoDiffVecXd timestep_i = q.tail(1);

  *y = getDynamicsConstraint(s_i, s_iplus1, timestep_i, theta_sDDot_);
}

void DynamicsConstraint::DoEval(const
                                Eigen::Ref<const VectorX<Variable>>& x,
                                VectorX<Expression>*y) const {
  throw std::logic_error(
    "This constraint class does not support symbolic evaluation.");
}


VectorXd DynamicsConstraint::getGradientWrtTheta(
  const VectorXd & s_i, const VectorXd & s_iplus1,
  const VectorXd & timestep_i) const {
  VectorXd gradient(n_feature_sDDot_);
  for (int i = 0; i < n_feature_sDDot_; i++) {
    VectorXd theta_unit = VectorXd::Zero(theta_sDDot_.size());
    theta_unit(i) = 1;
    gradient(i) = getDynamicsConstraint(
      s_i, s_iplus1, timestep_i, theta_unit)(0) - s_iplus1(0) + s_i(0);
  }
  // TODO(yminchen): You can also use autoDiff to get the gradient herre.
  return gradient;
}



AutoDiffVecXd DynamicsConstraint::getDynamicsConstraint(
  const AutoDiffVecXd & s_i, const AutoDiffVecXd & s_iplus1,
  const AutoDiffVecXd & timestep_i, const VectorXd & theta) const {

  // Let the dynamics be dzdt = h(z;theta_sDDot) = h(z).
  AutoDiffVecXd h_of_s_i = expression_object_.getExpression(theta,
                           s_i);
  AutoDiffVecXd h_of_s_iplus1 = expression_object_.getExpression(theta,
                                s_iplus1);

  // Collocation point
  AutoDiffVecXd z_collocation = (s_i + s_iplus1) / 2 +
                                (h_of_s_i - h_of_s_iplus1) * timestep_i(0) / 8;

  AutoDiffVecXd h_of_colloc_pt = expression_object_.getExpression(theta,
                                 z_collocation);

  return (s_iplus1 - s_i) - timestep_i(0) * (h_of_s_i + 4 * h_of_colloc_pt +
         h_of_s_iplus1) / 6;
}
VectorXd DynamicsConstraint::getDynamicsConstraint(
  const VectorXd & s_i, const VectorXd & s_iplus1,
  const VectorXd & timestep_i, const VectorXd & theta) const {

  // Let the dynamics be dzdt = h(z;theta_sDDot) = h(z).
  VectorXd h_of_s_i =  expression_object_.getExpression(theta,
                       s_i);
  VectorXd h_of_s_iplus1 = expression_object_.getExpression(theta,
                           s_iplus1);

  // Collocation point
  VectorXd z_collocation = (s_i + s_iplus1) / 2 +
                           (h_of_s_i - h_of_s_iplus1) * timestep_i(0) / 8;

  VectorXd h_of_colloc_pt = expression_object_.getExpression(theta,
                            z_collocation);

  return (s_iplus1 - s_i) - timestep_i(0) * (h_of_s_i + 4 * h_of_colloc_pt +
         h_of_s_iplus1) / 6;
}



}  // namespace goldilocks_models
}  // namespace dairlib

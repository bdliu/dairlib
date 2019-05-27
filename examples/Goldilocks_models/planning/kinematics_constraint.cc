#include "examples/Goldilocks_models/planning/kinematics_constraint.h"


namespace dairlib {
namespace goldilocks_models {
namespace planning {

KinematicsConstraint::KinematicsConstraint(
  int n_r, int n_q, int n_feature_kin,
  const VectorXd & theta_kin,
  const std::string& description):
  Constraint(2 * n_r,
             2 * (n_r + n_q),
             VectorXd::Zero(2 * n_r),
             VectorXd::Zero(2 * n_r),
             description),
  n_r_(n_r),
  n_y_(2 * n_r),
  n_q_(n_q),
  n_x_(2 * n_q),
  theta_kin_(theta_kin),
  kin_expression_(KinematicsExpression<AutoDiffXd>(n_r, n_feature_kin)) {

  // Check the theta size
  DRAKE_DEMAND(n_r * n_feature_kin == theta_kin.size());

  // Check the feature size implemented in the model expression
  AutoDiffVecXd q_temp = initializeAutoDiff(VectorXd::Ones(n_q));
  DRAKE_DEMAND(n_feature_kin == kin_expression_.getFeature(q_temp).size());
}


void KinematicsConstraint::DoEval(const Eigen::Ref<const Eigen::VectorXd>& q,
                                  Eigen::VectorXd* y) const {
  AutoDiffVecXd y_t;
  Eval(initializeAutoDiff(q), &y_t);
  *y = autoDiffToValueMatrix(y_t);
}

void KinematicsConstraint::DoEval(const Eigen::Ref<const AutoDiffVecXd>& yx,
                                  AutoDiffVecXd* value) const {
  // Extract elements
  AutoDiffVecXd r = yx.segment(0, n_r_);
  AutoDiffVecXd dr = yx.segment(n_r_, n_r_);
  AutoDiffVecXd q = yx.segment(n_y_, n_q_);
  AutoDiffVecXd v = yx.segment(n_y_ + n_q_, n_q_);

  AutoDiffVecXd h = kin_expression_.getExpression(theta_kin_, q);

  VectorXd dhdt0 = autoDiffToGradientMatrix(h).block(0, n_y_, n_r_, n_q_) *
                   DiscardGradient(v);
  MatrixXd grad_dhdt = MatrixXd::Zero(n_r_, n_y_ + n_x_);
  for (int i = n_y_; i < n_y_ + n_q_; i++) {
    // Forward differencing
    q(i - n_y_) += 1e-8;
    VectorXd dhdti = autoDiffToGradientMatrix(
                       kin_expression_.getExpression(theta_kin_, q)).
                     block(0, n_y_, n_r_, n_q_) * DiscardGradient(v);
    q(i - n_y_) -= 1e-8;
    grad_dhdt.col(i) = (dhdti - dhdt0) / 1e-8;
  }
  grad_dhdt.block(0, n_y_ + n_q_, n_r_, n_q_) =
    autoDiffToGradientMatrix(h).block(0, n_y_, n_r_, n_q_);

  AutoDiffVecXd dhdt = initializeAutoDiff(dhdt0);
  drake::math::initializeAutoDiffGivenGradientMatrix(
    dhdt0, grad_dhdt, dhdt);

  VectorX<AutoDiffXd> output(n_y_);
  output << r - h,
         dr - dhdt;

  // Impose dynamics constraint
  *value = output;
}

void KinematicsConstraint::DoEval(const
                                  Eigen::Ref<const VectorX<Variable>>& x,
                                  VectorX<Expression>*y) const {
  throw std::logic_error(
    "This constraint class does not support symbolic evaluation.");
}



}  // namespace planning
}  // namespace goldilocks_models
}  // namespace dairlib

#include "examples/Goldilocks_models/planning/kinematics_constraint.h"


namespace dairlib {
namespace goldilocks_models {
namespace planning {

KinematicsConstraint::KinematicsConstraint(
  int n_r, int n_q, int n_feature_kin,
  const VectorXd & theta_kin,
  const std::string& description):
  Constraint(n_r,
             n_r + n_q,
             VectorXd::Zero(n_r),
             VectorXd::Zero(n_r),
             description),
  n_r_(n_r),
  n_q_(n_q),
  theta_kin_(theta_kin),
  kin_expression_(KinematicsExpression<AutoDiffXd>(n_r, n_feature_kin)) {

  // Check the theta size
  DRAKE_DEMAND(n_r * n_feature_kin == theta_kin.size());

  // Check the feature size implemented in the model expression
  VectorXd q_temp = VectorXd::Ones(n_q);
  DRAKE_DEMAND(n_feature_kin == kin_expression_.getFeature(q_temp).size());
}


void KinematicsConstraint::DoEval(const
                                Eigen::Ref<const Eigen::VectorXd>& q,
                                Eigen::VectorXd* y) const {
  AutoDiffVecXd y_t;
  Eval(initializeAutoDiff(q), &y_t);
  *y = autoDiffToValueMatrix(y_t);
}

void KinematicsConstraint::DoEval(const
                                Eigen::Ref<const AutoDiffVecXd>& rq,
                                AutoDiffVecXd* y) const {
  // Extract elements
  AutoDiffVecXd r = rq.head(n_r_);
  AutoDiffVecXd q = rq.tail(n_q_);

  // Impose dynamics constraint
  *y = r - kin_expression_.getExpression(theta_kin_, q);
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

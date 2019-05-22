#include "examples/Goldilocks_models/find_models/dynamics_constraint.h"


namespace dairlib {
namespace goldilocks_models {
namespace planning {

KinematicsConstraint::KinematicsConstraint(
  int n_r, int n_ddr, int n_feature_dyn,
  const VectorXd & theta_dyn,
  int n_tau,
  MatrixXd B_tau,
  const std::string& description):
  Constraint(2 * n_r,
             2 * (2 * n_r + n_tau) + 1,
             VectorXd::Zero(2 * n_r),
             VectorXd::Zero(2 * n_r),
             description),
  n_r_(n_r),
  n_ddr_(n_ddr),
  n_y_(n_r + n_ddr),
  n_feature_dyn_(n_feature_dyn),
  n_theta_dyn_(theta_dyn.size()),
  theta_dyn_(theta_dyn),
  n_tau_(n_tau),
  dyn_expression_(DynamicsExpression(n_ddr, n_feature_dyn, B_tau)) {

  // Check the theta size
  DRAKE_DEMAND(n_ddr * n_feature_dyn == theta_dyn.size());

  // Check the feature size implemented in the model expression
  VectorXd r_temp = VectorXd::Zero(n_ddr);
  VectorXd dr_temp = VectorXd::Zero(n_ddr);
  DRAKE_DEMAND(n_feature_dyn ==
               dyn_expression_.getFeature(r_temp, dr_temp).size());
}


void KinematicsConstraint::DoEval(const
                                Eigen::Ref<const Eigen::VectorXd>& q,
                                Eigen::VectorXd* y) const {
  AutoDiffVecXd y_t;
  Eval(initializeAutoDiff(q), &y_t);
  *y = autoDiffToValueMatrix(y_t);
}

void KinematicsConstraint::DoEval(const
                                Eigen::Ref<const AutoDiffVecXd>& ytyth,
                                AutoDiffVecXd* y) const {
  // Extract elements
  AutoDiffVecXd y_i = ytyth.head(n_y_);
  AutoDiffVecXd tau_i = ytyth.segment(n_y_, n_tau_);
  AutoDiffVecXd y_iplus1 = ytyth.segment(n_y_ + n_tau_, n_y_);
  AutoDiffVecXd tau_iplus1 = ytyth.segment(2 * (n_y_) + n_tau_, n_tau_);
  const AutoDiffVecXd h_i = ytyth.tail(1);

  // Impose dynamics constraint
  *y = getConstraintValueInAutoDiff(y_i, tau_i, y_iplus1, tau_iplus1, h_i);
}

void KinematicsConstraint::DoEval(const
                                Eigen::Ref<const VectorX<Variable>>& x,
                                VectorX<Expression>*y) const {
  throw std::logic_error(
    "This constraint class does not support symbolic evaluation.");
}

AutoDiffVecXd KinematicsConstraint::getConstraintValueInAutoDiff(
  const AutoDiffVecXd & y_i, const AutoDiffVecXd & tau_i,
  const AutoDiffVecXd & y_iplus1, const AutoDiffVecXd & tau_iplus1,
  const AutoDiffVecXd & h_i) const {
  //
  AutoDiffVecXd g_i = g(y_i, tau_i);
  AutoDiffVecXd g_iplus1 = g(y_iplus1, tau_iplus1);

  // Value of the cubic spline at collocation point
  AutoDiffVecXd y_c = (y_i + y_iplus1) / 2 + (g_i - g_iplus1) * h_i / 8;
  AutoDiffVecXd tau_c = (tau_i + tau_iplus1) / 2;

  // Get constraint value in autoDiff
  return (y_iplus1 - y_i) / h_i - (g_i + 4 * g(y_c, tau_c) + g_iplus1) / 6;
}

AutoDiffVecXd KinematicsConstraint::g(const AutoDiffVecXd & y,
                                    const AutoDiffVecXd & tau) const {
  AutoDiffVecXd dy = initializeAutoDiff(VectorXd::Zero(2 * n_r_));
  dy << y.tail(n_r_), dyn_expression_.getExpression(
       theta_dyn_, y.head(n_r_), y.tail(n_r_), tau);
  return dy;
}



}  // namespace planning
}  // namespace goldilocks_models
}  // namespace dairlib

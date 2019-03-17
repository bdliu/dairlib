#include "examples/Goldilocks_models/dynamics_constraint.h"


namespace dairlib {
namespace goldilocks_models {

DynamicsConstraint::DynamicsConstraint(
  int n_sDDot, int n_feature_sDDot,
  const VectorXd & theta_sDDot,
  const MultibodyPlant<AutoDiffXd> * plant,
  bool is_head,
  const std::string& description):
  Constraint(n_sDDot,
             2 * (plant->num_positions() + plant->num_velocities()) + 1,
             VectorXd::Zero(n_sDDot),
             VectorXd::Zero(n_sDDot),
             description),
  plant_(plant),
  n_q_(plant->num_positions()),
  n_v_(plant->num_velocities()),
  n_sDDot_(n_sDDot),
  n_feature_sDDot_(n_feature_sDDot),
  theta_sDDot_(theta_sDDot),
  expression_object_(DynamicsExpression(n_sDDot, n_feature_sDDot)),
  is_head_(is_head) {

  // Check the theta size
  DRAKE_DEMAND(n_sDDot * n_feature_sDDot == theta_sDDot.size());

  // Check the feature size implemented in the model expression
  VectorXd s_temp = VectorXd::Zero(n_sDDot);
  VectorXd ds_temp = VectorXd::Zero(n_sDDot);
  DRAKE_DEMAND(n_feature_sDDot ==
               expression_object_.getFeature(s_temp, ds_temp).size());
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
  const AutoDiffVecXd q_i = q.head(n_q_);
  const AutoDiffVecXd v_i = q.segment(n_q_, n_v_);
  const AutoDiffVecXd q_iplus1 = q.segment(n_q_ + n_v_, n_q_);
  const AutoDiffVecXd v_iplus1 = q.segment(2 * n_q_ + n_v_, n_v_);
  const AutoDiffVecXd timestep_i = q.tail(1);

  // Write a function get_s_ds() that gives you s_i, s_iplus1, ds_i, ds_iplus1
  // When you want to get B, you can call get_s_ds() and then
  // call the feature function.
  // Before you call get_s_ds(), you will need to format them into the DoEval
  // input here (i.e. autodiff q), so that you only need to call one function.
  // Also, you can call get_s_ds() to record your reduced order model state.

  // Maybe I should combine the two dynamics constraints into one, by a flag
  // difference, so that you don't have the duplicate get_s_ds() code.

  AutoDiffVecXd s_i;
  AutoDiffVecXd ds_i;
  AutoDiffVecXd s_iplus1;
  AutoDiffVecXd ds_iplus1;
  get_s_ds(q_i, v_i, s_i, ds_i);
  get_s_ds(q_iplus1, v_iplus1, s_iplus1, ds_iplus1);



  *y = initializeAutoDiff(VectorXd::Zero(n_sDDot_));
}

void DynamicsConstraint::DoEval(const
                                      Eigen::Ref<const VectorX<Variable>>& x,
                                      VectorX<Expression>*y) const {
  throw std::logic_error(
    "This constraint class does not support symbolic evaluation.");
}

void DynamicsConstraint::get_s_ds(
  const VectorXd & q_i, const VectorXd & dq_i,
  VectorXd & s_i, VectorXd & ds_i) const {
  // This is jsut for getting the double s_i and ds_i. (e.g. you want to record
  // it, etc.)

  // 1. initialize the autodiff yourself, so that it matches the format of the
  //      autodiffversion of get_s_ds.
  // 2. call the autodiffversion of get_s_ds
  // 3. discard the autodiff part
}
void DynamicsConstraint::get_s_ds(
  const AutoDiffVecXd & q_i, const AutoDiffVecXd & dq_i,
  AutoDiffVecXd & s_i, AutoDiffVecXd & ds_i) const {

  // AutoDiffVecXd h_of_z_i = expression_object_.getExpression(theta_sDDot_,
  //     z_i);
}

VectorXd DynamicsConstraint::getGradientWrtTheta(
  const VectorXd & s_i, const VectorXd & s_iplus1,
  const VectorXd & timestep_i) const {
  // TODO(yminchen): You need to use autoDiff to get the gradient here, because
  // it's a nonlinear function in theta.
  // The calculation here will not be the same as the one in eval(), because
  // we have totally different autodiff, and the second autodiff requires
  // costumization.

  // You'll need to create autoDiff yourself first, cause the input is double
  // and you need to jacobian to get ds.
  VectorXd gradient(n_feature_sDDot_);
  for (int i = 0; i < n_feature_sDDot_; i++) {
    VectorXd theta_unit = VectorXd::Zero(theta_sDDot_.size());
    theta_unit(i) = 1;
    gradient(i) = 0;
    // gradient(i) = getDynamicsConstraint(
    //   s_i, s_iplus1, timestep_i, theta_unit)(0) - s_iplus1(0) + s_i(0);
  }
  return gradient;
}




}  // namespace goldilocks_models
}  // namespace dairlib

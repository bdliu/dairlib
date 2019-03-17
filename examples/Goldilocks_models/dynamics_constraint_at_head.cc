#include "examples/Goldilocks_models/dynamics_constraint_at_head.h"


namespace dairlib {
namespace goldilocks_models {

DynamicsConstraintAtHead::DynamicsConstraintAtHead(
  int n_sDDot, int n_feature_sDDot,
  const VectorXd & theta_sDDot,
  const MultibodyPlant<AutoDiffXd> * plant,
  bool is_head,
  const std::string& description):
  Constraint(n_sDDot,
             2 * n_sDDot + 1,
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


void DynamicsConstraintAtHead::DoEval(const
                                      Eigen::Ref<const Eigen::VectorXd>& q,
                                      Eigen::VectorXd* y) const {
  AutoDiffVecXd y_t;
  Eval(initializeAutoDiff(q), &y_t);
  *y = autoDiffToValueMatrix(y_t);
}

void DynamicsConstraintAtHead::DoEval(const
                                      Eigen::Ref<const AutoDiffVecXd>& q,
                                      AutoDiffVecXd* y) const {
  const AutoDiffVecXd s_i = q.head(n_sDDot_);
  const AutoDiffVecXd s_iplus1 = q.segment(n_sDDot_, n_sDDot_);
  const AutoDiffVecXd timestep_i = q.tail(1);

  // Write a function get_s_ds() that gives you s_i, s_iplus1, ds_i, ds_iplus1
  // When you want to get B, you can call get_s_ds() and then
  // call the feature function.
  // Before you call get_s_ds(), you will need to format them into the DoEval
  // input here (i.e. autodiff q), so that you only need to call one function.
  // Also, you can call get_s_ds() to record your reduced order model state.

  // Maybe I should combine the two dynamics constraints into one, by a flag
  // difference, so that you don't have the duplicate get_s_ds() code.

  // AutoDiffVecXd hs_i = expression_object_



  // Let the dynamics be dzdt = h(z;theta_sDDot) = h(z).
  AutoDiffVecXd h_of_s_i = expression_object_.getExpression(theta_sDDot_,
                           s_i, s_i);
  AutoDiffVecXd h_of_s_iplus1 = expression_object_.getExpression(theta_sDDot_,
                                s_iplus1, s_iplus1);

  // Collocation point
  AutoDiffVecXd z_collocation = (s_i + s_iplus1) / 2 +
                                (h_of_s_i - h_of_s_iplus1) * timestep_i(0) / 8;

  AutoDiffVecXd h_of_colloc_pt = expression_object_.getExpression(theta_sDDot_,
                                 z_collocation, z_collocation);

  *y = (s_iplus1 - s_i) - timestep_i(0) * (h_of_s_i + 4 * h_of_colloc_pt +
       h_of_s_iplus1) / 6;
}

void DynamicsConstraintAtHead::DoEval(const
                                      Eigen::Ref<const VectorX<Variable>>& x,
                                      VectorX<Expression>*y) const {
  throw std::logic_error(
    "This constraint class does not support symbolic evaluation.");
}

VectorXd DynamicsConstraintAtHead::getGradientWrtTheta(
  const VectorXd & s_i, const VectorXd & s_iplus1,
  const VectorXd & timestep_i) const {
  // Just give back the feature.
  // You'll need to create autoDiff yourself, cause you need to jacobian to get ds.

  VectorXd gradient(n_feature_sDDot_);
  for (int i = 0; i < n_feature_sDDot_; i++) {
    VectorXd theta_unit = VectorXd::Zero(theta_sDDot_.size());
    theta_unit(i) = 1;
    gradient(i) = 0;
    // gradient(i) = getDynamicsConstraint(
    //   s_i, s_iplus1, timestep_i, theta_unit)(0) - s_iplus1(0) + s_i(0);
  }
  // TODO(yminchen): You can also use autoDiff to get the gradient herre.
  return gradient;
}




}  // namespace goldilocks_models
}  // namespace dairlib

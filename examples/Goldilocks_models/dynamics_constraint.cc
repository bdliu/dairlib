#include "examples/Goldilocks_models/dynamics_constraint.h"


namespace dairlib {
namespace goldilocks_models {

DynamicsConstraint::DynamicsConstraint(
  int n_s, int n_feature_s,
  const VectorXd & theta_s,
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
  n_s_(n_s),
  n_feature_s_(n_feature_s),
  theta_s_(theta_s),
  n_sDDot_(n_sDDot),
  n_feature_sDDot_(n_feature_sDDot),
  theta_sDDot_(theta_sDDot),
  kin_expression_(KinematicsExpression<AutoDiffXd>(n_s, n_feature_s)),
  dyn_expression_(DynamicsExpression(n_sDDot, n_feature_sDDot)),
  is_head_(is_head) {

  // Check the theta size
  DRAKE_DEMAND(n_s * n_feature_s == theta_s.size());

  // Check the feature size implemented in the model expression
  AutoDiffVecXd q_temp =
    initializeAutoDiff(VectorXd::Zero(plant->num_positions()));
  DRAKE_DEMAND(n_feature_s == kin_expression_.getFeature(q_temp).size());

  // Check the theta size
  DRAKE_DEMAND(n_sDDot * n_feature_sDDot == theta_sDDot.size());

  // Check the feature size implemented in the model expression
  VectorXd s_temp = VectorXd::Zero(n_sDDot);
  VectorXd ds_temp = VectorXd::Zero(n_sDDot);
  DRAKE_DEMAND(n_feature_sDDot ==
               dyn_expression_.getFeature(s_temp, ds_temp).size());
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
  // const AutoDiffVecXd q_i = q.head(n_q_);
  // const AutoDiffVecXd v_i = q.segment(n_q_, n_v_);
  AutoDiffVecXd x_i = q.head(n_q_+n_v_);
  // const AutoDiffVecXd q_iplus1 = q.segment(n_q_ + n_v_, n_q_);
  // const AutoDiffVecXd v_iplus1 = q.segment(2 * n_q_ + n_v_, n_v_);
  AutoDiffVecXd x_iplus1 = q.segment(n_q_ + n_v_, n_q_ + n_v_);
  const AutoDiffVecXd timestep_i = q.tail(1);

  // Write a function getSAndSDot() that gives you s_i, s_iplus1, ds_i, ds_iplus1
  // When you want to get B, you can call getSAndSDot() and then
  // call the feature function.
  // Before you call getSAndSDot(), you will need to format them into the DoEval
  // input here (i.e. autodiff q), so that you only need to call one function.
  // Also, you can call getSAndSDot() to record your reduced order model state.

  // Maybe I should combine the two dynamics constraints into one, by a flag
  // difference, so that you don't have the duplicate getSAndSDot() code.

  // VectorX<T> s_i(n_s_);
  AutoDiffVecXd s_i = initializeAutoDiff(VectorXd::Zero(n_s_));
  AutoDiffVecXd ds_i = initializeAutoDiff(VectorXd::Zero(n_s_));
  AutoDiffVecXd s_iplus1 = initializeAutoDiff(VectorXd::Zero(n_s_));
  AutoDiffVecXd ds_iplus1 = initializeAutoDiff(VectorXd::Zero(n_s_));
  getSAndSDot(x_i, s_i, ds_i, 0);
  getSAndSDot(x_iplus1, s_iplus1, ds_iplus1, n_q_ + n_v_);



  *y = initializeAutoDiff(VectorXd::Zero(n_sDDot_));
}

void DynamicsConstraint::DoEval(const
                                Eigen::Ref<const VectorX<Variable>>& x,
                                VectorX<Expression>*y) const {
  throw std::logic_error(
    "This constraint class does not support symbolic evaluation.");
}

void DynamicsConstraint::getSAndSDot(
  VectorXd x,
  VectorXd & s, VectorXd & ds, int i_start) const {
  // This is jsut for getting the double s_i and ds_i. (e.g. you want to record
  // it, etc.)

  // 1. initialize the autodiff yourself, so that it matches the format of the
  //      autodiffversion of getSAndSDot.
  // 2. call the autodiffversion of getSAndSDot
  // 3. discard the autodiff part
}
void DynamicsConstraint::getSAndSDot(
  AutoDiffVecXd x,
  AutoDiffVecXd & s, AutoDiffVecXd & ds, int i_start) const {
  AutoDiffVecXd q = x.head(n_q_);
  AutoDiffVecXd v = x.tail(n_v_);
  // s
  s = kin_expression_.getExpression(theta_s_, q);

  // ds
  MatrixXd d_phi0_d_q = autoDiffToGradientMatrix(
      kin_expression_.getFeature(q)).block(
          0, i_start, n_feature_s_, n_q_);
  VectorXd v0_val = DiscardGradient(x.tail(n_v_));
  VectorXd dphi0_dt = d_phi0_d_q * v0_val;

  MatrixXd grad_dphidt = MatrixXd::Zero(n_feature_s_, 2*(n_q_+n_v_)+1);
  for(int i = i_start; i<i_start+n_q_+n_v_; i++){
    x(i) += dx_;
    AutoDiffVecXd q = x.head(n_q_);

    MatrixXd d_phii_d_q = autoDiffToGradientMatrix(
        kin_expression_.getFeature(q)).block(
            0, i_start, n_feature_s_, n_q_);
    VectorXd vi_val = DiscardGradient(x.tail(n_v_));
    VectorXd dphii_dt = d_phii_d_q * vi_val;
    grad_dphidt.col(i) = (dphii_dt-dphi0_dt)/dx_;

    x(i) -= dx_;
  }

  cout << "dphi0_dt = \n" << dphi0_dt << endl;
  cout << "grad_dphidt = \n" << grad_dphidt << endl;

  AutoDiffVecXd dphi_dt = initializeAutoDiff(dphi0_dt);
  // VectorX<AutoDiffXd> dphi_dt(n_feature_s_);
  cout << "dphi_dt = " << dphi_dt << endl;
  cout << "autoDiffToGradientMatrix(dphi_dt) = " << autoDiffToGradientMatrix(dphi_dt) << endl;
  drake::math::initializeAutoDiffGivenGradientMatrix(dphi0_dt, grad_dphidt, dphi_dt);
  cout << "dphi_dt = " << dphi_dt<< endl;
  cout << "autoDiffToGradientMatrix(dphi_dt) = " << autoDiffToGradientMatrix(dphi_dt) << endl;

  for (int i = 0; i < n_s_ ; i++){
    // cout << "i = " << i <<endl;
    // cout << "theta_s_ = " << theta_s_.transpose() << endl;
    // cout << "n_feature_s_ = " << n_feature_s_ << endl;
    // cout << "theta_s_.segment(0, n_feature_s_) = " << theta_s_.segment(0, n_feature_s_) << endl;
    ds(i) = theta_s_.segment(i * n_feature_s_, n_feature_s_).dot(dphi_dt);
    // cout << "here\n";
  }
  cout << "ds = " << ds.transpose() << endl;

  cout << "outside for loop\n";
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

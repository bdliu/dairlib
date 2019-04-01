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
  n_theta_s_(theta_s.size()),
  theta_s_(theta_s),
  n_sDDot_(n_sDDot),
  n_feature_sDDot_(n_feature_sDDot),
  n_theta_sDDot_(theta_sDDot.size()),
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
                                Eigen::Ref<const AutoDiffVecXd>& qvqvh,
                                AutoDiffVecXd* y) const {
  // Extract elements
  AutoDiffVecXd x_i = qvqvh.head(n_q_ + n_v_);
  AutoDiffVecXd x_iplus1 = qvqvh.segment(n_q_ + n_v_, n_q_ + n_v_);
  const AutoDiffVecXd h_i = qvqvh.tail(1);

  // Impose dynamics constraint
  *y = getConstraintValueInAutoDiff(x_i, x_iplus1, h_i, theta_s_, theta_sDDot_);
}

void DynamicsConstraint::DoEval(const
                                Eigen::Ref<const VectorX<Variable>>& x,
                                VectorX<Expression>*y) const {
  throw std::logic_error(
    "This constraint class does not support symbolic evaluation.");
}


AutoDiffVecXd DynamicsConstraint::getConstraintValueInAutoDiff(
  const AutoDiffVecXd & x_i, const AutoDiffVecXd & x_iplus1,
  const AutoDiffVecXd & h_i,
  const VectorXd & theta_s, const VectorXd & theta_sDDot) const {

  // Get s and ds at knot i and i+1
  AutoDiffVecXd s_i = initializeAutoDiff(VectorXd::Zero(n_s_));
  AutoDiffVecXd ds_i = initializeAutoDiff(VectorXd::Zero(n_s_));
  AutoDiffVecXd s_iplus1 = initializeAutoDiff(VectorXd::Zero(n_s_));
  AutoDiffVecXd ds_iplus1 = initializeAutoDiff(VectorXd::Zero(n_s_));
  getSAndSDotInAutoDiff(x_i, s_i, ds_i, 0, theta_s);
  getSAndSDotInAutoDiff(x_iplus1, s_iplus1, ds_iplus1, n_q_ + n_v_, theta_s);

  // Get constraint value in autoDiff
  if (is_head_) {
    AutoDiffVecXd lhs =
      2 * (-3 * (s_i - s_iplus1) - h_i(0) * (ds_iplus1 + 2 * ds_i)) /
      (h_i(0) * h_i(0));

    AutoDiffVecXd rhs =
      dyn_expression_.getExpression(theta_sDDot, s_i, ds_i);

    return lhs - rhs;
  }
  else {
    AutoDiffVecXd lhs =
      (6 * (s_i - s_iplus1) + h_i(0) * (4 * ds_iplus1 + 2 * ds_i)) /
      (h_i(0) * h_i(0));

    AutoDiffVecXd rhs =
      dyn_expression_.getExpression(theta_sDDot, s_iplus1, ds_iplus1);

    return lhs - rhs;
  }
}


void DynamicsConstraint::getSAndSDotInAutoDiff(AutoDiffVecXd x,
    AutoDiffVecXd & s, AutoDiffVecXd & ds,
    const int & i_start,
    const VectorXd & theta_s) const {
  AutoDiffVecXd q = x.head(n_q_);
  AutoDiffVecXd v = x.tail(n_v_);
  // s
  s = kin_expression_.getExpression(theta_s, q);

  // ds
  MatrixXd d_phi0_d_q = autoDiffToGradientMatrix(kin_expression_.getFeature(q)).
                        block(0, i_start, n_feature_s_, n_q_);
  VectorXd v0_val = DiscardGradient(x.tail(n_v_));
  VectorXd dphi0_dt = d_phi0_d_q * v0_val;

  MatrixXd grad_dphidt = MatrixXd::Zero(n_feature_s_, 2 * (n_q_ + n_v_) + 1);
  for (int i = 0; i < n_q_ + n_v_; i++) {
    x(i) += eps_;
    AutoDiffVecXd q = x.head(n_q_);

    MatrixXd d_phii_d_q =
      autoDiffToGradientMatrix(
        kin_expression_.getFeature(q)).block(0, i_start, n_feature_s_, n_q_);
    VectorXd vi_val = DiscardGradient(x.tail(n_v_));
    VectorXd dphii_dt = d_phii_d_q * vi_val;
    grad_dphidt.col(i_start + i) = (dphii_dt - dphi0_dt) / eps_;

    x(i) -= eps_;
  }

  AutoDiffVecXd dphi_dt = initializeAutoDiff(dphi0_dt);
  drake::math::initializeAutoDiffGivenGradientMatrix(
    dphi0_dt, grad_dphidt, dphi_dt);

  for (int i = 0; i < n_s_ ; i++) {
    ds(i) = theta_s.segment(i * n_feature_s_, n_feature_s_).dot(dphi_dt);
  }
}


void DynamicsConstraint::getSAndSDot(
  const VectorXd & x,
  VectorXd & s, VectorXd & ds) const {
  // This is jsut for getting the double version of s and ds. (e.g. you want
  // to record it, etc.)
  // What we are doing here are:
  // 1. initialize the autodiff yourself, so that it matches the format of the
  //      autodiff version of getSAndSDot.
  // 2. call the autodiff version of getSAndSDot
  // 3. discard the gradient part

  AutoDiffVecXd x_autoDiff = initializeAutoDiff(x);
  AutoDiffVecXd s_autoDiff = initializeAutoDiff(VectorXd::Zero(n_s_));
  AutoDiffVecXd ds_autoDiff = initializeAutoDiff(VectorXd::Zero(n_s_));
  getSAndSDotInAutoDiff(x_autoDiff, s_autoDiff, ds_autoDiff, 0, theta_s_);

  s = DiscardGradient(s_autoDiff);
  ds = DiscardGradient(ds_autoDiff);
}

MatrixXd DynamicsConstraint::getGradientWrtTheta(
  const VectorXd & x_i_double, const VectorXd & x_iplus1_double,
  const VectorXd & h_i_double) const {
  // It's a nonlinear function in theta, so we use autoDiff to get the gradient.
  // The calculation here will not be the same as the one in eval(), because
  // we have totally different autodiff, and the second autodiff requires
  // costumization.
  // Also, since they are nonilnear, each row might have different gradient.

  // You'll need to create autoDiff yourself first, cause the input is double
  // and you need to jacobian to get ds.

  // ////////// V1: Do forward differencing on theta ///////////////////////////
  /*VectorXd q_i = x_i_double.head(n_q_);
  VectorXd v_i = x_i_double.tail(n_v_);
  VectorXd q_iplus1 = x_iplus1_double.head(n_q_);
  VectorXd v_iplus1 = x_iplus1_double.tail(n_v_);

  // Get s_i, ds_i
  AutoDiffVecXd q_i_ad = initializeAutoDiff(q_i);
  AutoDiffVecXd s_i_ad = kin_expression_.getExpression(theta_s_, q_i_ad);
  VectorXd s_i = autoDiffToValueMatrix(s_i_ad);
  VectorXd ds_i = autoDiffToGradientMatrix(s_i_ad) * v_i;
  // Get s_iplus1, ds_iplus1
  AutoDiffVecXd q_iplus1_ad = initializeAutoDiff(q_iplus1);
  AutoDiffVecXd s_iplus1_ad = kin_expression_.getExpression(
                                theta_s_, q_iplus1_ad);
  VectorXd s_iplus1 = autoDiffToValueMatrix(s_iplus1_ad);
  VectorXd ds_iplus1 = autoDiffToGradientMatrix(s_iplus1_ad) * v_iplus1;
  // Get constraint value
  VectorXd y_0;
  if (is_head_) {
    y_0 =
      2 * (-3 * (s_i - s_iplus1 ) - h_i_double(0) * (ds_iplus1 + 2 * ds_i)) /
      (h_i_double(0) * h_i_double(0)) -
      dyn_expression_.getExpression(theta_sDDot_, s_i, ds_i);
  }
  else {
    y_0 =
      (6 * (s_i - s_iplus1 ) + h_i_double(0) * (4 * ds_iplus1 + 2 * ds_i)) /
      (h_i_double(0) * h_i_double(0)) -
      dyn_expression_.getExpression(theta_sDDot_, s_iplus1, ds_iplus1);
  }

  // Get the gradient wrt theta_s and theta_sDDot
  VectorXd theta(n_theta_s_ + n_theta_sDDot_);
  theta << theta_s_, theta_sDDot_;
  MatrixXd gradWrtTheta(n_s_, theta.size());
  for (int k = 0; k < theta.size(); k++) {
    theta(k) += eps_;

    VectorXd theta_s = theta.head(n_theta_s_);
    VectorXd theta_sDDot = theta.tail(n_theta_sDDot_);

    // Get s_i, ds_i
    AutoDiffVecXd s_i_ad = kin_expression_.getExpression(theta_s, q_i_ad);
    VectorXd s_i = autoDiffToValueMatrix(s_i_ad);
    VectorXd ds_i = autoDiffToGradientMatrix(s_i_ad) * v_i;
    // Get s_iplus1, ds_iplus1
    AutoDiffVecXd s_iplus1_ad = kin_expression_.getExpression(
                                  theta_s, q_iplus1_ad);
    VectorXd s_iplus1 = autoDiffToValueMatrix(s_iplus1_ad);
    VectorXd ds_iplus1 = autoDiffToGradientMatrix(s_iplus1_ad) * v_iplus1;
    // Get constraint value
    VectorXd y_1;
    if (is_head_) {
      y_1 =
        2 * (-3 * (s_i - s_iplus1 ) - h_i_double(0) * (ds_iplus1 + 2 * ds_i)) /
        (h_i_double(0) * h_i_double(0)) -
        dyn_expression_.getExpression(theta_sDDot, s_i, ds_i);
    }
    else {
      y_1 =
        (6 * (s_i - s_iplus1 ) + h_i_double(0) * (4 * ds_iplus1 + 2 * ds_i)) /
        (h_i_double(0) * h_i_double(0)) -
        dyn_expression_.getExpression(theta_sDDot, s_iplus1, ds_iplus1);
    }

    // Get gradient
    gradWrtTheta.col(k) = (y_1 - y_0) / eps_;

    theta(k) -= eps_;
  }*/


  // ////////// V2: Do central differencing on theta ///////////////////////////
  // Get x_i, x_iplus1 and h_i in autoDiff
  VectorXd qvqvh_double(2 * (n_q_ + n_v_) + 1);
  qvqvh_double << x_i_double, x_iplus1_double, h_i_double;
  AutoDiffVecXd qvqvh = initializeAutoDiff(qvqvh_double);

  AutoDiffVecXd x_i = qvqvh.head(n_q_ + n_v_);
  AutoDiffVecXd x_iplus1 = qvqvh.segment(n_q_ + n_v_, n_q_ + n_v_);
  const AutoDiffVecXd h_i = qvqvh.tail(1);

  // Get the gradient wrt theta_s and theta_sDDot
  VectorXd theta(n_theta_s_ + n_theta_sDDot_);
  theta << theta_s_, theta_sDDot_;
  MatrixXd gradWrtTheta(n_s_, theta.size());
  std::vector<double> shift_vec{ -eps_ / 2, eps_ / 2};
  std::vector<VectorXd> y_vec;
  for (int k = 0; k < theta.size(); k++) {
    for (double shift : shift_vec) {
      theta(k) += shift;

      VectorXd theta_s = theta.head(n_theta_s_);
      VectorXd theta_sDDot = theta.tail(n_theta_sDDot_);

      // Evaluate constraint value
      y_vec.push_back(autoDiffToValueMatrix(getConstraintValueInAutoDiff(
                                              x_i, x_iplus1, h_i,
                                              theta_s, theta_sDDot)));

      theta(k) -= shift;
    }

    // Get gradient
    gradWrtTheta.col(k) = (y_vec[1] - y_vec[0]) / eps_;
    y_vec.clear();
  }

  return gradWrtTheta;
}






}  // namespace goldilocks_models
}  // namespace dairlib

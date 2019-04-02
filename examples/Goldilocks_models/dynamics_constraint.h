#pragma once

#include <string>

#include "drake/common/drake_throw.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/snopt_solver.h"
#include "multibody/multibody_utils.h"

#include "drake/common/drake_assert.h"
#include "examples/Goldilocks_models/kinematics_expression.h"
#include "examples/Goldilocks_models/dynamics_expression.h"

using std::map;
using std::string;
using std::vector;
using std::list;
using std::unique_ptr;
using std::make_unique;
using std::make_shared;
using std::isnan;
using std::isinf;

using Eigen::Dynamic;
using Eigen::AutoDiffScalar;
using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::Matrix;
using Eigen::MatrixXd;
using drake::VectorX;
using drake::MatrixX;
using drake::AutoDiffVecXd;
using drake::AutoDiffXd;
using drake::math::DiscardGradient;
using drake::math::autoDiffToValueMatrix;
using drake::math::autoDiffToGradientMatrix;
using drake::math::initializeAutoDiff;
using drake::solvers::to_string;
using drake::solvers::VectorXDecisionVariable;
using drake::solvers::Constraint;
using drake::solvers::MathematicalProgram;
using drake::solvers::Constraint;
using drake::solvers::VariableRefList;
using drake::solvers::Binding;
using drake::symbolic::Variable;
using drake::symbolic::Expression;
using drake::multibody::MultibodyPlant;


namespace dairlib {
namespace goldilocks_models {

class DynamicsConstraint : public Constraint {
 public:
  DynamicsConstraint(int n_s, int n_feature_s,
                     const VectorXd & theta_s,
                     int n_sDDot, int n_feature_sDDot,
                     const VectorXd & theta_sDDot,
                     const MultibodyPlant<AutoDiffXd> * plant,
                     bool is_head,
                     const std::string& description = "");
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& q,
              Eigen::VectorXd* y) const override;

  void DoEval(const Eigen::Ref<const drake::AutoDiffVecXd>& qvqvh,
              drake::AutoDiffVecXd* y) const override;

  void DoEval(const Eigen::Ref<const VectorX<Variable>>& q,
              VectorX<Expression>*y) const override;

  void getSAndSDot(const VectorXd & x,
                   VectorXd & s, VectorXd & ds) const;

  VectorXd getSDDot(const VectorXd & s, const VectorXd & ds) const {
    return dyn_expression_.getExpression(theta_sDDot_, s, ds);
  };

  MatrixXd getGradientWrtTheta(const VectorXd & x_i_double,
                               const VectorXd & x_iplus1_double,
                               const VectorXd & h_i_double) const;

  VectorXd getDynFeatures(const VectorXd & s, const VectorXd & ds) const {
    return dyn_expression_.getFeature(s, ds);
  };


 private:
  AutoDiffVecXd getConstraintValueInAutoDiff(
    const AutoDiffVecXd & x_i, const AutoDiffVecXd & x_iplus1,
    const AutoDiffVecXd & h_i,
    const VectorXd & theta_s, const VectorXd & theta_sDDot) const;

  void getSAndSDotInAutoDiff(AutoDiffVecXd x_i,
                             AutoDiffVecXd & s_i,
                             AutoDiffVecXd & ds_i,
                             const int & i_start,
                             const VectorXd & theta_s) const;

  const MultibodyPlant<AutoDiffXd> * plant_;
  int n_q_;
  int n_v_;
  int n_s_;
  int n_feature_s_;
  int n_theta_s_;
  VectorXd theta_s_;
  int n_sDDot_;
  int n_feature_sDDot_;
  int n_theta_sDDot_;
  VectorXd theta_sDDot_;
  KinematicsExpression<AutoDiffXd> kin_expression_;
  DynamicsExpression dyn_expression_;
  bool is_head_;
  double eps_fd_ = 1e-6;
  double eps_cd_ = 1e-4;
  double eps_ho_ = 1e-3;
  // The above number is tested in getGradientWrtTheta(), and the result is:
  // 1e-6 good for fd
  // 1e-4 good for cd;
  // 1e-3 good for ho;
  vector<double> fd_shift_vec_{0, eps_fd_};  // forward difference
  vector<double> cd_shift_vec_{ -eps_cd_ / 2, eps_cd_ / 2};  // central difference
  vector<double> ho_shift_vec_{ -eps_ho_ / 2, -eps_ho_ / 4,
                                eps_ho_ / 4, eps_ho_ / 2};
};
}  // namespace goldilocks_models
}  // namespace dairlib

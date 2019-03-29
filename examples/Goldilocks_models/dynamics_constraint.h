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

  void getSAndSDot(VectorXd x,
                   VectorXd & s, VectorXd & ds) const;
  void getSAndSDot(AutoDiffVecXd x_i,
                   AutoDiffVecXd & s_i, AutoDiffVecXd & ds_i, int i_start) const;

  VectorXd getSDDot(VectorXd s, VectorXd ds) const;

  MatrixXd getGradientWrtTheta(
  VectorXd theta_s, VectorXd theta_sDDot,
  const VectorXd & x_i, const VectorXd & x_iplus1,
  const VectorXd & h_i) const;

 private:
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
  double eps_ = 1e-8;
};
}  // namespace goldilocks_models
}  // namespace dairlib

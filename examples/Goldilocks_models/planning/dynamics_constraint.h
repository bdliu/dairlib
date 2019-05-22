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
namespace planning {


class DynamicsConstraint : public Constraint {
 public:
  DynamicsConstraint(int n_r, int n_ddr, int n_feature_dyn,
                     const VectorXd & theta_dyn,
                     int n_tau,
                     MatrixXd B_tau,
                     const std::string& description = "");
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& q,
              Eigen::VectorXd* y) const override;

  void DoEval(const Eigen::Ref<const drake::AutoDiffVecXd>& ytyth,
              drake::AutoDiffVecXd* y) const override;

  void DoEval(const Eigen::Ref<const VectorX<Variable>>& q,
              VectorX<Expression>*y) const override;

 private:
  AutoDiffVecXd getConstraintValueInAutoDiff(
    const AutoDiffVecXd & y_i, const AutoDiffVecXd & tau_i,
    const AutoDiffVecXd & y_iplus1, const AutoDiffVecXd & tau_iplus1,
    const AutoDiffVecXd & h_i) const;

  // y = [r; dr]
  // Calculate the dynamics for y
  AutoDiffVecXd g(const AutoDiffVecXd & y, const AutoDiffVecXd & tau) const;

  int n_r_;
  int n_ddr_;
  int n_feature_dyn_;
  int n_theta_ddr_;
  VectorXd theta_ddr_;
  int n_y_;
  int n_tau_;
  DynamicsExpression dyn_expression_;

  double eps_fd_ = 1e-6;
  double eps_cd_ = 1e-4;
  double eps_ho_ = 1e-3;
  // The above number is tested in getGradientWrtTheta(), and the result is:
  // 1e-6 good for fd
  // 1e-4 good for cd;  // B matrix error ~ 1e-13 to 1e-15
  // 1e-3 good for ho;
  vector<double> fd_shift_vec_{0, eps_fd_};  // forward difference
  vector<double> cd_shift_vec_{ -eps_cd_ / 2, eps_cd_ / 2};  // central difference
  vector<double> ho_shift_vec_{ -eps_ho_ / 2, -eps_ho_ / 4,
                                eps_ho_ / 4, eps_ho_ / 2};
};
}  // namespace planning
}  // namespace goldilocks_models
}  // namespace dairlib

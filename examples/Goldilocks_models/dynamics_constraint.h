#pragma once

#include <string>

#include "drake/common/drake_throw.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/snopt_solver.h"
#include "multibody/multibody_utils.h"

#include "drake/common/drake_assert.h"
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
  DynamicsConstraint(int n_sDDot, int n_feature_sDDot, VectorXd & theta_sDDot,
                     const MultibodyPlant<AutoDiffXd> * plant,
                     const std::string& description = "");
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& q,
              Eigen::VectorXd* y) const override;

  void DoEval(const Eigen::Ref<const drake::AutoDiffVecXd>& q,
              drake::AutoDiffVecXd* y) const override;

  void DoEval(const Eigen::Ref<const VectorX<Variable>>& q,
              VectorX<Expression>*y) const override;

  VectorXd getGradientWrtTheta(
      const VectorXd & s_i, const VectorXd & s_iplus1,
      const VectorXd & timestep_i) const;

  AutoDiffVecXd getDynamicsConstraint(
      const AutoDiffVecXd & s_i, const AutoDiffVecXd & s_iplus1,
      const AutoDiffVecXd & timestep_i, const VectorXd & theta) const;
  VectorXd getDynamicsConstraint(
      const VectorXd & s_i, const VectorXd & s_iplus1,
      const VectorXd & timestep_i, const VectorXd & theta) const;


 private:
  const MultibodyPlant<AutoDiffXd> * plant_;
  int n_sDDot_;
  int n_feature_sDDot_;
  VectorXd theta_sDDot_;
  DynamicsExpression expression_object_;
};
}  // namespace goldilocks_models
}  // namespace dairlib

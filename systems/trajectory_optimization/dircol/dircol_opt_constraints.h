#pragma once

#include <memory.h>
#include <string>
#include <vector>
#include "drake/common/drake_copyable.h"
#include "drake/solvers/constraint.h"
#include "drake/common/symbolic.h"
#include "drake/systems/trajectory_optimization/multiple_shooting.h"
#include "systems/trajectory_optimization/dircol/dircol_kinematic_data.h"
#include "systems/trajectory_optimization/dircol/dircol_kinematic_data_set.h"

namespace dairlib {
namespace systems {
namespace trajectory_optimization {

/// Helper class for all dircol constraints
/// manages evaluation of functions and numerical gradients
template <typename T>
class DircolAbstractConstraint : public drake::solvers::Constraint {
 public:
  DircolAbstractConstraint(int num_constraints, int num_vars,
                           const Eigen::VectorXd& lb,
                           const Eigen::VectorXd& ub,
                           const std::string& description = "");

 public:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override;

  void DoEval(const Eigen::Ref<const drake::AutoDiffVecXd>& x,
              drake::AutoDiffVecXd* y) const override;

  void DoEval(
      const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>>&,
      drake::VectorX<drake::symbolic::Expression>*) const override;

  virtual void EvaluateConstraint(const Eigen::Ref<const drake::VectorX<T>>& x,
              drake::VectorX<T>* y) const = 0;
};

enum DircolKinConstraintType { kPosOnly = 1/*, kNothing = 0*/ };
// Note that you implement constraint C17&C18 of Ayonga's paper in
// impactconstraint, which includes post-impact velocity. You should take out
// C17 if you want to expand DircolKinConstraintType to include kVel.


template <typename T>
class CubicInterpolationConstraint : public DircolAbstractConstraint<T> {
 public:
//  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CubicInterpolationConstraint)

 public:
  CubicInterpolationConstraint(const drake::multibody::MultibodyPlant<T>& plant,
                          DircolKinematicDataSet<T>& constraints);

  ~CubicInterpolationConstraint() override = default;

  int num_states() const { return num_states_; }
  int num_inputs() const { return num_inputs_; }

 public:
  void EvaluateConstraint(const Eigen::Ref<const drake::VectorX<T>>& x,
                          drake::VectorX<T>* y) const override;

 private:
  CubicInterpolationConstraint(const drake::multibody::MultibodyPlant<T>& plant,
                          DircolKinematicDataSet<T>& constraints,
                          int num_positions, int num_velocities, int num_inputs);

  const drake::multibody::MultibodyPlant<T>& plant_;
  DircolKinematicDataSet<T>* constraints_;
  const int num_states_{0};
  const int num_inputs_{0};
  const int num_kinematic_constraints_{0};
  const int num_positions_{0};
  const int num_velocities_{0};
  const int num_xv_{0};
  const int num_xvu_{0};
};


/// Implements the direct collocation constraints for a first-order hold on
/// the input and a cubic polynomial representation of the state trajectories.
/// This class is based on the similar constraint used by DirectCollocation,
/// but incorporates the effect of constraint forces

template <typename T>
class DircolDynamicConstraint : public DircolAbstractConstraint<T> {
 public:
//  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DircolDynamicConstraint)

 public:
  DircolDynamicConstraint(const drake::multibody::MultibodyPlant<T>& plant,
                          DircolKinematicDataSet<T>& constraints,
                          bool is_floating_base = false);

  ~DircolDynamicConstraint() override = default;

  int num_states() const { return num_states_; }
  int num_inputs() const { return num_inputs_; }
  int num_kinematic_constraints() const { return num_kinematic_constraints_; }

 public:
  void EvaluateConstraint(const Eigen::Ref<const drake::VectorX<T>>& x,
                          drake::VectorX<T>* y) const override;

 private:
  DircolDynamicConstraint(const drake::multibody::MultibodyPlant<T>& plant,
                          DircolKinematicDataSet<T>& constraints,
                          int num_positions, int num_velocities, int num_inputs,
                          int num_kinematic_constraints, int num_quat_slack);

  const drake::multibody::MultibodyPlant<T>& plant_;
  DircolKinematicDataSet<T>* constraints_;
  const int num_states_{0};
  const int num_inputs_{0};
  const int num_kinematic_constraints_{0};
  const int num_positions_{0};
  const int num_velocities_{0};
  const int num_quat_slack_{0};
};


/// Implements the kinematic constraints used by Dircol
/// For constraints given by c(q), enforces the three constraints
///   c(q), d/dt c(q), d^2/dt^2 c(q)
///   at the position, velocity, and constraint levels
/// When used with states that are fully constrained, i.e. if q_0 or x_0
/// is fully specified, this class contains the option DircolKinConstraintType
/// to only include a subset of the three constraints above, avoiding
/// redundancy.
///
/// Constraints may also be specified as relative, where rather than c(q)=0,
/// we have the constriant c(q)=constant. The constant value is a then new
/// optimization decision variable.
template <typename T>
class DircolKinematicConstraint : public DircolAbstractConstraint<T> {
 public:
  /// Constructor. Defaults the relative constraints to be all false
  /// @param plant the MultibodyPlant
  /// @param DircolKinematicDataSet the set of kinematic constraints
  /// @param type the constraint type. All (default), accel and vel, accel only.
  DircolKinematicConstraint(const drake::multibody::MultibodyPlant<T>& plant,
    DircolKinematicDataSet<T>& constraint_data,
    DircolKinConstraintType type = DircolKinConstraintType::kPosOnly);
  /// Constructor
  /// @param plant the MultibodyPlant
  /// @param DircolKinematicDataSet the set of kinematic constraints
  /// @param is_constraint_relative vector of booleans
  /// @param type the constraint type. All (default), accel and vel, accel only.
  DircolKinematicConstraint(const drake::multibody::MultibodyPlant<T>& plant,
    DircolKinematicDataSet<T>& constraint_data,
    std::vector<bool> is_constraint_relative,
    drake::VectorX<double> phi_vals,
    DircolKinConstraintType type = DircolKinConstraintType::kPosOnly);

  ~DircolKinematicConstraint() override = default;

  void EvaluateConstraint(const Eigen::Ref<const drake::VectorX<T>>& x,
                          drake::VectorX<T>* y) const override;

 private:
  DircolKinematicConstraint(const drake::multibody::MultibodyPlant<T>& plant,
                            DircolKinematicDataSet<T>& constraint_data,
                            std::vector<bool> is_constraint_relative,
                            drake::VectorX<double> phi_vals,
                            DircolKinConstraintType type, int num_positions,
                            int num_velocities, int num_inputs,
                            int num_kinematic_constraints);


  const drake::multibody::MultibodyPlant<T>& plant_;
  DircolKinematicDataSet<T>* constraints_;

  const int num_states_{0};
  const int num_inputs_{0};
  const int num_kinematic_constraints_{0};
  const int num_positions_{0};
  const int num_velocities_{0};
  const DircolKinConstraintType type_{kPosOnly};
  const std::vector<bool> is_constraint_relative_;
  const drake::VectorX<double> phi_vals_;
  const int n_relative_;
  Eigen::MatrixXd relative_map_;
};


/// Implements the hybrid impact constraints used by Dircol
/// Enforces the impact constraint that vp = vm + M^{-1}*J^T*Lambda
template <typename T>
class DircolImpactConstraint : public DircolAbstractConstraint<T> {
 public:
  /// @param plant the MultibodyPlant
  /// @param DircolKinematicDataSet the set of kinematic constraints
  DircolImpactConstraint(const drake::multibody::MultibodyPlant<T>& plant,
                         DircolKinematicDataSet<T>& constraint_data);

  ~DircolImpactConstraint() override = default;

  void EvaluateConstraint(const Eigen::Ref<const drake::VectorX<T>>& x,
                          drake::VectorX<T>* y) const override;

 private:
  DircolImpactConstraint(const drake::multibody::MultibodyPlant<T>& plant,
                         DircolKinematicDataSet<T>& constraint_data,
                         int num_positions, int num_velocities,
                         int num_kinematic_constraints);


  const drake::multibody::MultibodyPlant<T>& plant_;
  DircolKinematicDataSet<T>* constraints_;

  const int num_states_{0};
  const int num_kinematic_constraints_{0};
  const int num_positions_{0};
  const int num_velocities_{0};
};


/// Helper method to add a DircolDynamicConstraint to the @p prog,
/// ensuring that the order of variables in the binding matches the order
/// expected by the constraint.
// Note: The order of arguments is a compromise between GSG and the desire to
// match the AddConstraint interfaces in MathematicalProgram.
//
// mposa: I don't think this function is actually being used, and I'm not sure what it does
template <typename T>
drake::solvers::Binding<drake::solvers::Constraint> AddDircolConstraint(
    std::shared_ptr<DircolDynamicConstraint<T>> constraint,
    const Eigen::Ref<const drake::solvers::VectorXDecisionVariable>& timestep,
    const Eigen::Ref<const drake::solvers::VectorXDecisionVariable>& state,
    const Eigen::Ref<const drake::solvers::VectorXDecisionVariable>& next_state,
    const Eigen::Ref<const drake::solvers::VectorXDecisionVariable>& input,
    const Eigen::Ref<const drake::solvers::VectorXDecisionVariable>& next_input,
    const Eigen::Ref<const drake::solvers::VectorXDecisionVariable>& force,
    const Eigen::Ref<const drake::solvers::VectorXDecisionVariable>& next_force,
    const Eigen::Ref<const drake::solvers::VectorXDecisionVariable>& collocation_force,
    const Eigen::Ref<const drake::solvers::VectorXDecisionVariable>& collocation_position_slack,
    drake::solvers::MathematicalProgram* prog);



}  // namespace trajectory_optimization
}  // namespace systems
}  // namespace dairlib

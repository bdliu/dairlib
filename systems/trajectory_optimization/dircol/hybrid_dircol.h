#pragma once

#include <memory.h>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/solvers/constraint.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/system.h"
#include "drake/systems/trajectory_optimization/multiple_shooting.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/common/symbolic.h"

#include "systems/trajectory_optimization/dircol/dircol_opt_constraints.h"
#include "systems/trajectory_optimization/dircol/dircol_options.h"
#include "systems/trajectory_optimization/dircol/dircol_kinematic_data.h"
#include "systems/trajectory_optimization/dircol/dircol_kinematic_data_set.h"

namespace dairlib {
namespace systems {
namespace trajectory_optimization {

/// DIRCOL implements the approach to trajectory optimization as
/// described in
///   A. Hereid, E. A. Cousineau, C. M. Hubicki, and A. D. Ames, “3D Dynamic
///   Walking With Underactuated Humanoid Robots: A Direct Collocation Framework
///   for Optimizing Hybrid Zero Dynamics,” in 2016 IEEE International Conference
///   on Robotics and Automation (ICRA), Stockholm, Sweden, 2016, pp. 1447–1454.
/// The code is modified from DIRCON.

template <typename T>
class HybridDircol :
    public drake::systems::trajectory_optimization::MultipleShooting {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(HybridDircol)

  /// Constructs the %MathematicalProgram% and adds the collocation constraints.
  ///
  /// @param plant The MultibodyPlant describing the plant and kinematics
  /// @param num_time_samples The number of knot points in the trajectory.
  /// @param minimum_timestep Minimum spacing between sample times.
  /// @param maximum_timestep Maximum spacing between sample times.
  /// @param constraints The set of kinematic constraints that must be enforced
  /// @param opttions (see DircolOptions)
  HybridDircol(const drake::multibody::MultibodyPlant<T>& plant,
               std::vector<int> num_time_samples,
               std::vector<double> minimum_timestep,
               std::vector<double> maximum_timestep,
               std::vector<DircolKinematicDataSet<T>*> constraints,
               std::vector<DircolOptions> options,
               bool is_quaterion = false);

  ~HybridDircol() override {}

  /// Get the input trajectory at the solution as a
  /// %drake::trajectories::PiecewisePolynomialTrajectory%.
  drake::trajectories::PiecewisePolynomial<double> ReconstructInputTrajectory(
      const drake::solvers::MathematicalProgramResult& result) const override;

  /// Get the state trajectory at the solution as a
  /// %drake::trajectories::PiecewisePolynomialTrajectory%.
  drake::trajectories::PiecewisePolynomial<double> ReconstructStateTrajectory(
      const drake::solvers::MathematicalProgramResult& result) const override;

  /// Set the initial guess for the force variables for a specific mode
  /// @param mode the mode index
  /// @param traj_init_l contact forces lambda (interpreted at knot points)
  /// @param traj_init_lc contact forces (interpretted at collocation points)
  /// @param traj_init_vc velocity constraint slack variables (at collocation)
  void SetInitialForceTrajectory(int mode,
      const drake::trajectories::PiecewisePolynomial<double>& traj_init_l,
      const drake::trajectories::PiecewisePolynomial<double>& traj_init_lc,
      const drake::trajectories::PiecewisePolynomial<double>& traj_init_vc);



  int num_kinematic_constraints(int mode) const {
    return num_kinematic_constraints_[mode];
  }

  const drake::solvers::VectorXDecisionVariable& dv_vars(int mode) const {
    return dv_vars_[mode]; }
  Eigen::VectorBlock<const drake::solvers::VectorXDecisionVariable> dv(
        int mode, int index) const {
    return dv_vars_[mode].segment(index * num_velocities_, num_velocities_);
  }

  const drake::solvers::VectorXDecisionVariable& force_vars(int mode) const {
    return force_vars_[mode];}
  Eigen::VectorBlock<const drake::solvers::VectorXDecisionVariable> force(
        int mode, int index) const {
    return force_vars_[mode].segment(index * num_kinematic_constraints_[mode],
                                     num_kinematic_constraints_[mode]);
  }

  const drake::solvers::VectorXDecisionVariable& offset_vars(int mode) const {
    return offset_vars_[mode];
  }

  const drake::solvers::VectorXDecisionVariable& x_col_vars(int mode)
      const { return x_col_vars_[mode]; }
  Eigen::VectorBlock<const drake::solvers::VectorXDecisionVariable> x_col(
        int mode, int index) const {
    return x_col_vars_[mode].segment(index * num_states(), num_states());
  }
  const drake::solvers::VectorXDecisionVariable& dv_col_vars(int mode)
      const { return dv_col_vars_[mode]; }
  Eigen::VectorBlock<const drake::solvers::VectorXDecisionVariable> dv_col(
        int mode, int index) const {
    return dv_col_vars_[mode].segment(index * num_velocities_, num_velocities_);
  }
  const drake::solvers::VectorXDecisionVariable& u_col_vars(int mode)
      const { return u_col_vars_[mode]; }
  Eigen::VectorBlock<const drake::solvers::VectorXDecisionVariable> u_col(
        int mode, int index) const {
    return u_col_vars_[mode].segment(index * num_inputs(), num_inputs());
  }
  const drake::solvers::VectorXDecisionVariable& force_col_vars(int mode)
      const { return force_col_vars_[mode]; }
  Eigen::VectorBlock<const drake::solvers::VectorXDecisionVariable> force_col(
        int mode, int index) const {
    return force_col_vars_[mode].segment(index * num_kinematic_constraints_[mode],
                                     num_kinematic_constraints_[mode]);
  }

  const drake::solvers::VectorXDecisionVariable& impulse_vars(int mode)
      const { return impulse_vars_[mode]; }

  const drake::solvers::VectorXDecisionVariable& v_post_impact_vars() const {
    return v_post_impact_vars_;
  }
  const Eigen::VectorBlock<const drake::solvers::VectorXDecisionVariable>
      v_post_impact_vars_by_mode(int mode) const;

  /// Get the state decision variables given a mode and a time_index
  /// (time_index is w.r.t that particular mode). This will use the
  ///  v_post_impact_vars_ if needed. Otherwise, it just returns the standard
  /// x_vars element
  drake::solvers::VectorXDecisionVariable state_vars_by_mode(int mode,
      int time_index) const;

  drake::VectorX<drake::symbolic::Expression> SubstitutePlaceholderVariables(
      const drake::VectorX<drake::symbolic::Expression>& f,
      int interval_index) const;

  using drake::systems::trajectory_optimization::MultipleShooting::N;
  using drake::systems::trajectory_optimization::MultipleShooting::
        SubstitutePlaceholderVariables;

 private:
  // Implements a running cost at all timesteps using trapezoidal integration.
  const drake::multibody::MultibodyPlant<T>& plant_;
  std::vector<DircolKinematicDataSet<T>*> constraints_;
  const int num_modes_;
  const std::vector<int> mode_lengths_;
  std::vector<int> mode_start_;
  void DoAddRunningCost(const drake::symbolic::Expression& e) override;
  std::vector<drake::solvers::VectorXDecisionVariable> x_col_vars_;
  std::vector<drake::solvers::VectorXDecisionVariable> dv_col_vars_;
  std::vector<drake::solvers::VectorXDecisionVariable> u_col_vars_;
  const drake::solvers::VectorXDecisionVariable v_post_impact_vars_;
  std::vector<drake::solvers::VectorXDecisionVariable> dv_vars_;
  std::vector<drake::solvers::VectorXDecisionVariable> force_vars_;
  std::vector<drake::solvers::VectorXDecisionVariable> force_col_vars_;
  std::vector<drake::solvers::VectorXDecisionVariable> offset_vars_;
  std::vector<drake::solvers::VectorXDecisionVariable> impulse_vars_;

  int num_velocities_;
  std::vector<int> num_kinematic_constraints_;
};

}  // namespace trajectory_optimization
}  // namespace systems
}  // namespace dairlib

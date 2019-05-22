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

namespace dairlib {
namespace goldilocks_models {

// Reduced order model
// y = [r; dr]
// ddr = theta_ddr * phi_ddr + B * tau

// Modified from HybridDircon class
class RomPlanningTrajOptWithFomImpactMap :
  public drake::systems::trajectory_optimization::MultipleShooting {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(RomPlanningTrajOptWithFomImpactMap)

  RomPlanningTrajOptWithFomImpactMap(vector<int> num_time_samples,
                                     vector<double> minimum_timestep,
                                     vector<double> maximum_timestep,
                                     int n_r,
                                     int n_tau,
                                     const VectorXd & theta_kin,
                                     const VectorXd & theta_dyn,
                                     const MultibodyPlant<double>& plant);

  ~RomPlanningTrajOptWithFomImpactMap() override {}

  /// TODO: remove when removed upstream
  drake::trajectories::PiecewisePolynomial<double> ReconstructInputTrajectory()
  const override {
    return drake::trajectories::PiecewisePolynomial<double>();
  };
  drake::trajectories::PiecewisePolynomial<double> ReconstructStateTrajectory()
  const override {
    return drake::trajectories::PiecewisePolynomial<double>();
  };

  /// Get the input trajectory at the solution as a
  /// %drake::trajectories::PiecewisePolynomialTrajectory%.
  drake::trajectories::PiecewisePolynomial<double> ReconstructInputTrajectory(
    const drake::solvers::MathematicalProgramResult& result) const override;

  /// Get the state trajectory at the solution as a
  /// %drake::trajectories::PiecewisePolynomialTrajectory%.
  drake::trajectories::PiecewisePolynomial<double> ReconstructStateTrajectory(
    const drake::solvers::MathematicalProgramResult& result) const override;

  const drake::solvers::VectorXDecisionVariable& dr_post_impact_vars() const {
    return dr_post_impact_vars_;
  }

  const Eigen::VectorBlock<const drake::solvers::VectorXDecisionVariable>
  dr_post_impact_vars_by_mode(int mode) const;

  /// Get the state decision variables given a mode and a time_index
  /// (time_index is w.r.t that particular mode). This will use the
  ///  dr_post_impact_vars_ if needed. Otherwise, it just returns the standard
  /// x_vars element
  drake::solvers::VectorXDecisionVariable state_vars_by_mode(int mode,
      int time_index) const;

  drake::VectorX<drake::symbolic::Expression> SubstitutePlaceholderVariables(
    const drake::VectorX<drake::symbolic::Expression>& f,
    int interval_index) const;

 private:
  // Implements a running cost at all timesteps using trapezoidal integration.
  void DoAddRunningCost(const drake::symbolic::Expression& e) override;
  const int num_modes_;
  const std::vector<int> mode_lengths_;
  std::vector<int> mode_start_;
  const drake::solvers::VectorXDecisionVariable dr_post_impact_vars_;
  const int n_s_;
  const int n_tau_;
  const drake::multibody::MultibodyPlant<double>& plant_;
};

}  // namespace goldilocks_models
}  // namespace dairlib

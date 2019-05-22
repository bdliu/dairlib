#include "examples/Goldilocks_models/planning/RoM_planning_traj_opt.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "drake/solvers/decision_variable.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"

#include "examples/Goldilocks_models/kinematics_expression.h"
#include "examples/Goldilocks_models/dynamics_expression.h"
#include "examples/Goldilocks_models/dynamics_constraint.h"

namespace dairlib {
namespace goldilocks_models {

using drake::solvers::Binding;
using drake::solvers::Constraint;
using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
using drake::solvers::VectorXDecisionVariable;
using drake::trajectories::PiecewisePolynomial;
using drake::systems::trajectory_optimization::MultipleShooting;
using drake::multibody::MultibodyPlant;
using drake::AutoDiffXd;
using drake::VectorX;
using drake::solvers::VectorXDecisionVariable;
using drake::symbolic::Expression;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

RomPlanningTrajOptWithFomImpactMap::RomPlanningTrajOptWithFomImpactMap(
  vector<int> num_time_samples,
  vector<double> minimum_timestep,
  vector<double> maximum_timestep,
  int n_r,
  int n_tau,
  const VectorXd & theta_kin,
  const VectorXd & theta_dyn,
  const MultibodyPlant<double>& plant) :
  MultipleShooting(n_tau,
                   2 * n_tau,
                   std::accumulate(num_time_samples.begin(),
                                   num_time_samples.end(), 0) - num_time_samples.size() + 1, 1e-8, 1e8),
  num_modes_(num_time_samples.size()),
  mode_lengths_(num_time_samples),
  ds_post_impact_vars_(NewContinuousVariables(n_r *
                       (num_time_samples.size() - 1), "v_p")),
  n_r_(n_r),
  n_tau_(n_tau),
  plant_(plant) {
  DRAKE_ASSERT(minimum_timestep.size() == num_modes_);
  DRAKE_ASSERT(maximum_timestep.size() == num_modes_);

  // Initialization is looped over the modes
  int counter = 0;
  for (int i = 0; i < num_modes_; i++) {
    mode_start_.push_back(counter);

    // set timestep bounds
    for (int j = 0; j < mode_lengths_[i] - 1; j++) {
      AddBoundingBoxConstraint(minimum_timestep[i], maximum_timestep[i],
                               timestep(mode_start_[i] + j));
    }
    for (int j = 0; j < mode_lengths_[i] - 2; j++) {
      // all timesteps must be equal
      AddLinearConstraint(timestep(mode_start_[i] + j) ==
                          timestep(mode_start_[i] + j + 1));
    }

    // Add dynamics constraints at collocation points
    auto dyn_constraint = std::make_shared<planning::DynamicConstraint<T>>(
                            n_r, n_ddr, n_feature_dyn, theta_dyn, n_tau, B_tau);
    DRAKE_ASSERT(
      static_cast<int>(dyn_constraint->num_constraints()) == num_states());
    for (int j = 0; j < mode_lengths_[i] - 1; j++) {
      int time_index = mode_start_[i] + j;
      AddConstraint(dyn_constraint,
      { state_vars_by_mode(i, j),
        u_vars().segment(time_index * num_inputs(), num_inputs()),
        state_vars_by_mode(i, j + 1),
        u_vars().segment((time_index + 1) * num_inputs(), num_inputs()),
        h_vars().segment(time_index, 1)
      });
    }

    // Add kinematics constraints

    // TODO: not finished yet
    auto kin_constraint = std::make_shared<planning::DynamicConstraint<T>>(
                            n_r, n_ddr, n_feature_dyn, theta_dyn, n_tau, B_tau);
    for (int j = 0; j < mode_lengths_[i]; j++) {
      int time_index = mode_start_[i] + j;
      AddConstraint(kin_constraint,
      { state_vars_by_mode(i, j),
        u_vars().segment(time_index * num_inputs(), num_inputs()),
        state_vars_by_mode(i, j + 1),
        u_vars().segment((time_index + 1) * num_inputs(), num_inputs()),
        h_vars().segment(time_index, 1)
      });
    }


    counter += mode_lengths_[i] - 1;
  }
}

const Eigen::VectorBlock<const VectorXDecisionVariable>
RomPlanningTrajOptWithFomImpactMap::ds_post_impact_vars_by_mode(
  int mode) const {
  return ds_post_impact_vars_.segment(mode * n_r_, n_r_);
}

VectorX<Expression>
RomPlanningTrajOptWithFomImpactMap::SubstitutePlaceholderVariables(
  const VectorX<Expression>& f, int interval_index) const {
  VectorX<Expression> ret(f.size());
  for (int i = 0; i < f.size(); i++) {
    ret(i) = MultipleShooting::SubstitutePlaceholderVariables(f(i),
             interval_index);
  }
  return ret;
}


// Eigen::VectorBlock<const VectorXDecisionVariable> RomPlanningTrajOptWithFomImpactMap::state_vars_by_mode(int mode, int time_index)  {
VectorXDecisionVariable RomPlanningTrajOptWithFomImpactMap::state_vars_by_mode(
  int mode, int time_index) const {
  if (time_index == 0 && mode > 0) {
    VectorXDecisionVariable ret(num_states());
    ret << x_vars().segment((mode_start_[mode] + time_index)*num_states(), n_r_),
        ds_post_impact_vars_by_mode(mode - 1);
    return ret;
    // return Eigen::VectorBlock<const VectorXDecisionVariable>(ret, 0, num_states());
  } else {
    VectorXDecisionVariable ret(num_states());
    return x_vars().segment((mode_start_[mode] + time_index) * num_states(),
                            num_states());
    // std::cout << Eigen::VectorBlock<VectorXDecisionVariable>(ret, 0, num_states())  << std::endl;
    // return Eigen::VectorBlock<VectorXDecisionVariable>(ret, 0, num_states());
  }
}

//TODO: need to configure this to handle the hybrid discontinuities properly
void RomPlanningTrajOptWithFomImpactMap::DoAddRunningCost(
  const drake::symbolic::Expression& g) {
  // Trapezoidal integration:
  //    sum_{i=0...N-2} h_i/2.0 * (g_i + g_{i+1}), or
  // g_0*h_0/2.0 + [sum_{i=1...N-2} g_i*(h_{i-1} + h_i)/2.0] +
  // g_{N-1}*h_{N-2}/2.0.

  AddCost(MultipleShooting::SubstitutePlaceholderVariables(g, 0) *
          h_vars()(0) / 2);
  for (int i = 1; i <= N() - 2; i++) {
    AddCost(MultipleShooting::SubstitutePlaceholderVariables(g , i) *
            (h_vars()(i - 1) + h_vars()(i)) / 2);
  }
  AddCost(MultipleShooting::SubstitutePlaceholderVariables(g, N() - 1) *
          h_vars()(N() - 2) / 2);
}

PiecewisePolynomial<double>
RomPlanningTrajOptWithFomImpactMap::ReconstructInputTrajectory(
  const MathematicalProgramResult& result) const {
  Eigen::VectorXd times = GetSampleTimes(result);
  vector<double> times_vec(N());
  vector<Eigen::MatrixXd> inputs(N());
  for (int i = 0; i < N(); i++) {
    times_vec[i] = times(i);
    inputs[i] = result.GetSolution(input(i));
  }
  return PiecewisePolynomial<double>::FirstOrderHold(times_vec, inputs);
}

PiecewisePolynomial<double>
RomPlanningTrajOptWithFomImpactMap::ReconstructStateTrajectory(
  const MathematicalProgramResult& result) const {
  VectorXd times_all(GetSampleTimes(result));
  VectorXd times(N() + num_modes_ - 1);

  MatrixXd states(num_states(), N() + num_modes_ - 1);
  MatrixXd inputs(num_inputs(), N() + num_modes_ - 1);
  MatrixXd derivatives(num_states(), N() + num_modes_ - 1);

  for (int i = 0; i < num_modes_; i++) {
    for (int j = 0; j < mode_lengths_[i]; j++) {
      int k = mode_start_[i] + j + i;
      int k_data = mode_start_[i] + j;
      times(k) = times_all(k_data);

      // False timestep to match velocities
      if (i > 0 && j == 0) {
        times(k) += + 1e-6;
      }
      VectorX<double> yk = result.GetSolution(state_vars_by_mode(i, j));
      VectorX<double> tauk = result.GetSolution(input(k_data));
      states.col(k) = yk;
      inputs.col(k) = tauk;


      // TODO(yminchen): need to modify the following code
      auto context = multibody::createContext(plant_, yk, tauk);
      constraints_[i]->updateData(*context, result.GetSolution(force(i, j)));
      derivatives.col(k) =
        drake::math::DiscardGradient(constraints_[i]->getXDot());
    }
  }
  return PiecewisePolynomial<double>::Cubic(times, states, derivatives);
}



}  // namespace goldilocks_models
}  // namespace dairlib

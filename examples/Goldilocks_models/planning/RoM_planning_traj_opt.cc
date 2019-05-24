#include "examples/Goldilocks_models/planning/RoM_planning_traj_opt.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
#include <string>

#include "drake/solvers/decision_variable.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"

#include "examples/Goldilocks_models/kinematics_expression.h"
#include "examples/Goldilocks_models/dynamics_expression.h"
#include "examples/Goldilocks_models/planning/kinematics_constraint.h"
#include "examples/Goldilocks_models/planning/dynamics_constraint.h"
#include "examples/Goldilocks_models/planning/FoM_guard_constraint.h"
// #include "examples/Goldilocks_models/planning/FoM_reset_map_constraint.h"
#include "examples/Goldilocks_models/planning/FoM_stance_foot_constraint.h"

namespace dairlib {
namespace goldilocks_models {

using drake::solvers::Binding;
using drake::solvers::Constraint;
using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
using drake::solvers::VectorXDecisionVariable;
using drake::trajectories::PiecewisePolynomial;
using drake::systems::trajectory_optimization::MultipleShooting;
using drake::AutoDiffXd;
using drake::VectorX;
using drake::solvers::VectorXDecisionVariable;
using drake::symbolic::Expression;

using std::string;
using std::to_string;

RomPlanningTrajOptWithFomImpactMap::RomPlanningTrajOptWithFomImpactMap(
  vector<int> num_time_samples,
  vector<double> minimum_timestep,
  vector<double> maximum_timestep,
  MatrixXd Q,
  MatrixXd R,
  int n_r,
  int n_tau,
  MatrixXd B_tau,
  int n_feature_kin,
  int n_feature_dyn,
  const VectorXd & theta_kin,
  const VectorXd & theta_dyn,
  const MultibodyPlant<double>& plant) :
  MultipleShooting(n_tau,
                   2 * n_r,
                   std::accumulate(num_time_samples.begin(),
                                   num_time_samples.end(), 0) - num_time_samples.size() + 1, 1e-8, 1e8),
  num_modes_(num_time_samples.size()),
  mode_lengths_(num_time_samples),
  dr_post_impact_vars_(NewContinuousVariables(
                         n_r * (num_time_samples.size() - 1), "dr_p")),
  x0_vars_(NewContinuousVariables(
             (plant.num_positions() + plant.num_velocities()) * num_time_samples.size(),
             "x0")),
  xf_vars_(NewContinuousVariables(
             (plant.num_positions() + plant.num_velocities()) * num_time_samples.size(),
             "xf")),
  n_r_(n_r),
  n_tau_(n_tau),
  n_x_(plant.num_positions() + plant.num_velocities()),
  plant_(plant) {
  DRAKE_ASSERT(minimum_timestep.size() == num_modes_);
  DRAKE_ASSERT(maximum_timestep.size() == num_modes_);

  map<string, int> positions_map = multibody::makeNameToPositionsMap(plant);

  // Parameters
  bool zero_touchdown_impact = true;
  double desired_final_position = 1;

  // Add cost
  cout << "Adding cost...\n";
  auto y = this->state();
  auto tau = this->input();
  this->AddRunningCost(tau.transpose()*R * tau);
  this->AddRunningCost(y.tail(n_r).transpose()*Q * y.tail(n_r));

  // (Constraint) Initialization is looped over the modes
  int counter = 0;
  for (int i = 0; i < num_modes_; i++) {
    cout << "Mode " << i << endl;
    mode_start_.push_back(counter);

    // Set timestep bounds
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
    cout << "Adding dynamics constraint...\n";
    auto dyn_constraint = std::make_shared<planning::DynamicsConstraint>(
                            n_r, n_r, n_feature_dyn, theta_dyn, n_tau, B_tau);
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
    cout << "Adding kinematics constraint...\n";
    bool left_stance = (i % 2 == 0) ? true : false;
    int n_q = plant_.num_positions();
    auto kin_constraint = std::make_shared<planning::KinematicsConstraint>(
                            left_stance, n_r, n_q, n_feature_kin, theta_kin);
    std::vector<int> j_vec{0, mode_lengths_[i] - 1};
    for (unsigned int k = 0; k < j_vec.size(); k++) {
      int j = j_vec[k];
      auto y_j = state_vars_by_mode(i, j);
      if (k == 0)
        AddConstraint(kin_constraint, {y_j.head(n_r_), x0_vars_by_mode(i).head(n_q)});
      else
        AddConstraint(kin_constraint, {y_j.head(n_r_), xf_vars_by_mode(i).head(n_q)});
    }

    // Add periodicity constraints
    if (i != 0) {
      cout << "Adding periodicity constraint...\n";
      AddLinearConstraint(xf_vars_by_mode(i - 1).segment(0, n_q) ==
                          x0_vars_by_mode(i).segment(0, n_q));
    }

    // Add guard constraint
    cout << "Adding guard constraint...\n";
    VectorXd lb = VectorXd::Zero(2);
    if (!zero_touchdown_impact)
      lb << 0, -std::numeric_limits<double>::infinity();
    VectorXd ub = VectorXd::Zero(2);
    auto guard_constraint = std::make_shared<planning::FomGuardConstraint>(
                              left_stance, n_q, n_q, lb, ub);
    AddConstraint(guard_constraint, xf_vars_by_mode(i));

    // Add reset map constraint
    if (i != 0) {
      cout << "Adding reset map constraint...\n";
      if (zero_touchdown_impact) {
        AddLinearConstraint(xf_vars_by_mode(i - 1).segment(n_q, n_q) ==
                            x0_vars_by_mode(i).segment(n_q, n_q));
      } else {
        /*int n_J = (zero_touchdown_impact) ? 0 : 2;
        auto reset_map_constraint =
          std::make_shared<planning::FomResetMapConstraint>(
            left_stance, n_q, n_q, n_J, plant_);
        auto Lambda = NewContinuousVariables(n_J, "Lambda" + to_string(i));
        AddConstraint(reset_map_constraint, {xf_vars_by_mode(i - 1),
                                             xf_vars_by_mode(i),
                                             Lambda
                                            });*/
      }
    }

    // Full order model joint limits
    cout << "Adding full-order model joint constraint...\n";
    vector<string> l_or_r{"left_", "right_"};
    vector<string> fom_joint_names{"hip_pin", "knee_pin"};
    vector<double> lb_for_fom_joints{ -M_PI / 2.0, 5.0 / 180.0 * M_PI};
    vector<double> ub_for_fom_joints{ M_PI / 2.0, M_PI / 2.0};
    for (unsigned int k = 0; k < l_or_r.size(); k++) {
      for (unsigned int l = 0; l < fom_joint_names.size(); l++) {
        AddLinearConstraint(
          x0_vars_by_mode(i)(positions_map.at(l_or_r[k] + fom_joint_names[l])),
          lb_for_fom_joints[l], ub_for_fom_joints[l]);
        AddLinearConstraint(
          xf_vars_by_mode(i)(positions_map.at(l_or_r[k] + fom_joint_names[l])),
          lb_for_fom_joints[l], ub_for_fom_joints[l]);
      }
    }

    // Sitching x0 and xf
    cout << "Adding full-order model stance foot constraint...\n";
    auto fom_sf_constraint = std::make_shared<planning::FomStanceFootConstraint>(
                               left_stance, n_q);
    AddConstraint(fom_sf_constraint, {x0_vars_by_mode(i).head(n_q),
                                      xf_vars_by_mode(i).head(n_q)
                                     });

    // Additional constraints for the full order model
    if (i == 0) {
      cout << "Adding initial position constraint for full-order model...\n";
      AddLinearConstraint(x0_vars_by_mode(i)(0) == 0);
    } else if (i == num_modes_ - 1) {
      cout << "Adding final position constraint for full-order model...\n";
      AddLinearConstraint(xf_vars_by_mode(i)(0) == desired_final_position);
    }

    counter += mode_lengths_[i] - 1;
  }
}

const Eigen::VectorBlock<const VectorXDecisionVariable>
RomPlanningTrajOptWithFomImpactMap::dr_post_impact_vars_by_mode(
  int mode) const {
  return dr_post_impact_vars_.segment(mode * n_r_, n_r_);
}
const Eigen::VectorBlock<const VectorXDecisionVariable>
RomPlanningTrajOptWithFomImpactMap::x0_vars_by_mode(int mode) const {
  return x0_vars_.segment(mode * n_x_, n_x_);
}
const Eigen::VectorBlock<const VectorXDecisionVariable>
RomPlanningTrajOptWithFomImpactMap::xf_vars_by_mode(int mode) const {
  return xf_vars_.segment(mode * n_x_, n_x_);
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
        dr_post_impact_vars_by_mode(mode - 1);
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
      /*auto context = multibody::createContext(plant_, yk, tauk);
      constraints_[i]->updateData(*context, result.GetSolution(force(i, j)));
      derivatives.col(k) =
        drake::math::DiscardGradient(constraints_[i]->getXDot());*/
    }
  }
  // return PiecewisePolynomial<double>::Cubic(times, states, derivatives);
  return PiecewisePolynomial<double>::FirstOrderHold(times, states);
}



}  // namespace goldilocks_models
}  // namespace dairlib

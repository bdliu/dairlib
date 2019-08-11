#include "hybrid_dircon.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "drake/solvers/decision_variable.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "multibody/multibody_utils.h"

namespace dairlib {
namespace systems {
namespace trajectory_optimization {

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

template <typename T>
HybridDircon<T>::HybridDircon(
    const MultibodyPlant<T>& plant,
    vector<int> num_time_samples,
    vector<double> minimum_timestep,
    vector<double> maximum_timestep,
    vector<DirconKinematicDataSet<T>*> constraints,
    vector<DirconOptions> options,
    bool is_quaterion) :
        MultipleShooting(plant.num_actuators(),
          plant.num_positions() + plant.num_velocities(),
          std::accumulate(num_time_samples.begin(),
          num_time_samples.end(), 0) - num_time_samples.size() + 1, 1e-8, 1e8),
    plant_(plant),
    constraints_(constraints),
    num_modes_(num_time_samples.size()),
    mode_lengths_(num_time_samples),
    v_post_impact_vars_(NewContinuousVariables(plant.num_velocities() *
                        (num_time_samples.size() - 1), "v_p")) {
  DRAKE_ASSERT(minimum_timestep.size() == num_modes_);
  DRAKE_ASSERT(maximum_timestep.size() == num_modes_);
  DRAKE_ASSERT(constraints.size() == num_modes_);


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

    // initialize constraint lengths
    num_kinematic_constraints_.push_back(constraints_[i]->countConstraints());

    // initialize decision variables
    force_vars_.push_back(NewContinuousVariables(constraints_[i]->countConstraints() * num_time_samples[i], "lambda[" + std::to_string(i) + "]"));
    collocation_force_vars_.push_back(NewContinuousVariables(constraints_[i]->countConstraints() * (num_time_samples[i] - 1), "lambda_c[" + std::to_string(i) + "]"));
    collocation_slack_vars_.push_back(NewContinuousVariables(constraints_[i]->countConstraints() * (num_time_samples[i] - 1), "v_c[" + std::to_string(i) + "]"));
    offset_vars_.push_back(NewContinuousVariables(options[i].getNumRelative(), "offset[" + std::to_string(i) + "]"));
    if (i > 0) {
      impulse_vars_.push_back(NewContinuousVariables(constraints_[i]->countConstraints(), "impulse[" + std::to_string(i) + "]"));
    }

    auto constraint = std::make_shared<DirconDynamicConstraint<T>>(plant_, *constraints_[i], is_quaterion);

    DRAKE_ASSERT(static_cast<int>(constraint->num_constraints()) == num_states());

    // For N-1 timesteps, add a constraint which depends on the knot
    // value along with the state and input vectors at that knot and the
    // next.

    //Adding dynamic constraints
    for (int j = 0; j < mode_lengths_[i] - 1; j++) {
      int time_index = mode_start_[i] + j;
      vector<VectorXDecisionVariable> x_next;

      // gamma is slack variable used to scale quaternion norm to 1.
      auto gamma = NewContinuousVariables((is_quaterion)? 1:0, "gamma_"+ std::to_string(i) + "_" + std::to_string(j));
      AddConstraint(constraint,
                    {h_vars().segment(time_index,1),
                     state_vars_by_mode(i, j),
                     state_vars_by_mode(i, j+1),
                     u_vars().segment(time_index * num_inputs(), num_inputs() * 2),
                     force_vars(i).segment(j * num_kinematic_constraints(i), num_kinematic_constraints(i) * 2),
                     collocation_force_vars(i).segment(j * num_kinematic_constraints(i), num_kinematic_constraints(i)),
                     collocation_slack_vars(i).segment(j * num_kinematic_constraints(i), num_kinematic_constraints(i)),
                     gamma});

    }

    // Adding kinematic constraints
    auto kinematic_constraint = std::make_shared<DirconKinematicConstraint<T>>(plant_, *constraints_[i],
      options[i].getConstraintsRelative(), options[i].getPhiValues());
    for (int j = 1; j < mode_lengths_[i] - 1; j++) {
      int time_index = mode_start_[i] + j;
      AddConstraint(kinematic_constraint,
                    {state_vars_by_mode(i,j),
                     u_vars().segment(time_index * num_inputs(), num_inputs()),
                     force_vars(i).segment(j * num_kinematic_constraints(i), num_kinematic_constraints(i)),
                     offset_vars(i)});
    }

    // special case first and last timestep based on options
    auto kinematic_constraint_start = std::make_shared<DirconKinematicConstraint<T>>(plant_, *constraints_[i],
      options[i].getConstraintsRelative(), options[i].getPhiValues(), options[i].getStartType());
    AddConstraint(kinematic_constraint_start,
                  {state_vars_by_mode(i,0),
                   u_vars().segment(mode_start_[i], num_inputs()),
                   force_vars(i).segment(0, num_kinematic_constraints(i)),
                   offset_vars(i)});

    if (mode_lengths_[i] > 1) {
      auto kinematic_constraint_end = std::make_shared<DirconKinematicConstraint<T>>(plant_, *constraints_[i],
        options[i].getConstraintsRelative(), options[i].getPhiValues(), options[i].getEndType());
      AddConstraint(kinematic_constraint_end,
                    {state_vars_by_mode(i, mode_lengths_[i] - 1),
                     u_vars().segment((mode_start_[i] + mode_lengths_[i] - 1) * num_inputs(), num_inputs()),
                     force_vars(i).segment((mode_lengths_[i]-1) * num_kinematic_constraints(i), num_kinematic_constraints(i)),
                     offset_vars(i)});
    }


    //Add constraints on force and impulse variables
    for (int l = 0; l < mode_lengths_[i] - 1; l++) {
      int start_index = l*num_kinematic_constraints(i);
      for (int j = 0; j < constraints_[i]->getNumConstraintObjects(); j++) {
        DirconKinematicData<T>* constraint_j = constraints_[i]->getConstraint(j);
        start_index += constraint_j->getLength();
        for (int k = 0; k < constraint_j->numForceConstraints(); k++) {
          AddConstraint(constraint_j->getForceConstraint(k), force_vars(i).segment(start_index, constraint_j->getLength()));
        }
      }
    }

    //Force cost option
    if (options[i].getForceCost() != 0) {
      auto A = options[i].getForceCost()*MatrixXd::Identity(num_kinematic_constraints(i),num_kinematic_constraints(i));
      auto b = MatrixXd::Zero(num_kinematic_constraints(i),1);
      for (int j=0; j <  mode_lengths_[i]; j++) {
        AddL2NormCost(A,b,force(i,j));
      }
    }

    if (i > 0) {
      if (num_kinematic_constraints(i) > 0) {
        auto impact_constraint = std::make_shared<DirconImpactConstraint<T>>(plant_, *constraints_[i]);
        AddConstraint(impact_constraint,
                {state_vars_by_mode(i-1, mode_lengths_[i-1] - 1), // last state from previous mode
                 impulse_vars(i-1),
                 v_post_impact_vars_by_mode(i-1)});

        //Add constraints on impulse variables
        int start_index = 0;
        for (int j = 0; j < constraints_[i]->getNumConstraintObjects(); j++) {
          DirconKinematicData<T>* constraint_j = constraints_[i]->getConstraint(j);
          for (int k = 0; k < constraint_j->numForceConstraints(); k++) {
            AddConstraint(constraint_j->getForceConstraint(k), impulse_vars(i-1).segment(start_index, constraint_j->getLength()));
          }
          start_index += constraint_j->getLength();
        }

      } else {
        auto x_vars_prev = state_vars_by_mode(i-1, mode_lengths_[i-1] - 1);
        AddConstraint(v_post_impact_vars_by_mode(i-1) == x_vars_prev.tail(plant.num_velocities()));
      }
    }

    counter += mode_lengths_[i] - 1;
  }
}

template <typename T>
const Eigen::VectorBlock<const VectorXDecisionVariable> HybridDircon<T>::v_post_impact_vars_by_mode(int mode) const {
  return v_post_impact_vars_.segment(mode * plant_.num_velocities(), plant_.num_velocities());
}

template <typename T>
VectorX<Expression> HybridDircon<T>::SubstitutePlaceholderVariables(
      const VectorX<Expression>& f, int interval_index) const {
  VectorX<Expression> ret(f.size());
  for (int i = 0; i < f.size(); i++) {
    ret(i) = MultipleShooting::SubstitutePlaceholderVariables(f(i),
                                                              interval_index);
  }
  return ret;
}


// Eigen::VectorBlock<const VectorXDecisionVariable> HybridDircon<T>::state_vars_by_mode(int mode, int time_index)  {
template <typename T>
VectorXDecisionVariable HybridDircon<T>::state_vars_by_mode(int mode, int time_index) const {
  if (time_index == 0 && mode > 0) {
    VectorXDecisionVariable ret(num_states());
    ret << x_vars().segment((mode_start_[mode] + time_index)*num_states(), plant_.num_positions()),
          v_post_impact_vars_by_mode(mode - 1);
    return ret;
    // return Eigen::VectorBlock<const VectorXDecisionVariable>(ret, 0, num_states());
  } else {
    VectorXDecisionVariable ret(num_states());
    return x_vars().segment((mode_start_[mode] + time_index)*num_states(), num_states());
    // std::cout << Eigen::VectorBlock<VectorXDecisionVariable>(ret, 0, num_states())  << std::endl;
    // return Eigen::VectorBlock<VectorXDecisionVariable>(ret, 0, num_states());
  }
}

//TODO: need to configure this to handle the hybrid discontinuities properly
template <typename T>
void HybridDircon<T>::DoAddRunningCost(const drake::symbolic::Expression& g) {
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

template <typename T>
PiecewisePolynomial<double> HybridDircon<T>::ReconstructInputTrajectory(
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

// TODO(mposa)
// need to configure this to handle the hybrid discontinuities properly
template <typename T>
PiecewisePolynomial<double> HybridDircon<T>::ReconstructStateTrajectory(
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
      VectorX<T> xk = result.GetSolution(state_vars_by_mode(i, j));
      VectorX<T> uk = result.GetSolution(input(k_data));
      states.col(k) = xk;
      inputs.col(k) = uk;
      auto context = multibody::createContext(plant_, xk, uk);
      constraints_[i]->updateData(*context, result.GetSolution(force(i, j)));
      derivatives.col(k) =
          drake::math::DiscardGradient(constraints_[i]->getXDot());
  }
}
  return PiecewisePolynomial<double>::Cubic(times, states, derivatives);
}

template <typename T>
void HybridDircon<T>::SetInitialForceTrajectory(int mode,
    const PiecewisePolynomial<double>& traj_init_l,
    const PiecewisePolynomial<double>& traj_init_lc,
    const PiecewisePolynomial<double>& traj_init_vc) {
  double start_time = 0;
  double h;
  if (timesteps_are_decision_variables())
    h = GetInitialGuess(h_vars()[0]);
  else
    h = fixed_timestep();

  VectorXd guess_force(force_vars_[mode].size());
  if (traj_init_l.empty()) {
    guess_force.fill(0);  // Start with 0
  } else {
    for (int i = 0; i < mode_lengths_[mode]; ++i) {
      guess_force.segment(num_kinematic_constraints_[mode] * i,
                          num_kinematic_constraints_[mode]) =
          traj_init_l.value(start_time + i * h);
    }
  }
  SetInitialGuess(force_vars_[mode], guess_force);

  VectorXd guess_collocation_force(collocation_force_vars_[mode].size());
  if (traj_init_lc.empty()) {
    guess_collocation_force.fill(0);  // Start with 0
  } else {
    for (int i = 0; i < mode_lengths_[mode]-1; ++i) {
      guess_collocation_force.segment(num_kinematic_constraints_[mode] * i,
                                      num_kinematic_constraints_[mode]) =
          traj_init_lc.value(start_time + (i + 0.5) * h);
    }
  }
  SetInitialGuess(collocation_force_vars_[mode], guess_collocation_force);

  VectorXd guess_collocation_slack(collocation_slack_vars_[mode].size());
  if (traj_init_vc.empty()) {
    guess_collocation_slack.fill(0);  // Start with 0
  } else {
    for (int i = 0; i < mode_lengths_[mode]-1; ++i) {
      guess_collocation_slack.segment(num_kinematic_constraints_[mode] * i,
                                      num_kinematic_constraints_[mode]) =
          traj_init_vc.value(start_time + (i + 0.5) * h);
    }
  }
  // call superclass method
  SetInitialGuess(collocation_slack_vars_[mode], guess_collocation_slack);
}

template class HybridDircon<double>;
// template class HybridDircon<AutoDiffXd>;

}  // namespace trajectory_optimization
}  // namespace systems
}  // namespace dairlib

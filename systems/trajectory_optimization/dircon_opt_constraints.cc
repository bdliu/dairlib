#include "dircon_opt_constraints.h"
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "multibody/multibody_utils.h"

#include "systems/goldilocks_models/file_utils.h"  // writeCSV

namespace dairlib {
namespace systems {
namespace trajectory_optimization {

using drake::multibody::MultibodyPlant;
using drake::systems::Context;
using drake::solvers::Binding;
using drake::solvers::Constraint;
using drake::solvers::MathematicalProgram;
using drake::solvers::VectorXDecisionVariable;
using drake::AutoDiffVecXd;
using drake::AutoDiffXd;
using drake::math::initializeAutoDiff;
using drake::math::autoDiffToValueMatrix;
using drake::VectorX;
using drake::MatrixX;
using Eigen::VectorXd;
using Eigen::MatrixXd;

template <typename T>
DirconAbstractConstraint<T>::DirconAbstractConstraint(int num_constraints, int num_vars,
                                                      const VectorXd& lb,
                                                      const VectorXd& ub,
                                                      const std::string& description)
  : Constraint(num_constraints, num_vars, lb, ub, description) {
}

template <>
void DirconAbstractConstraint<double>::DoEval(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    Eigen::VectorXd* y) const {
  EvaluateConstraint(x, y);
}

template <>
void DirconAbstractConstraint<AutoDiffXd>::DoEval(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    Eigen::VectorXd* y) const {
  AutoDiffVecXd y_t;
  EvaluateConstraint(initializeAutoDiff(x), &y_t);
  *y = autoDiffToValueMatrix(y_t);
}

template <typename T>
void DirconAbstractConstraint<T>::DoEval(
    const Eigen::Ref<const VectorX<drake::symbolic::Variable>>& x,
    VectorX<drake::symbolic::Expression>* y) const {
  throw std::logic_error(
      "DirconAbstractConstraint does not support symbolic evaluation.");
}

template <>
void DirconAbstractConstraint<AutoDiffXd>::DoEval(
    const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd* y) const {
  EvaluateConstraint(x,y);
}

template <>
void DirconAbstractConstraint<double>::DoEval(
    const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd* y) const {
    // forward differencing
    double dx = 1e-8;

    VectorXd x_val = autoDiffToValueMatrix(x);
    VectorXd y0,yi;
    EvaluateConstraint(x_val,&y0);

    MatrixXd dy = MatrixXd(y0.size(),x_val.size());
    for (int i=0; i < x_val.size(); i++) {
      x_val(i) += dx;
      EvaluateConstraint(x_val,&yi);
      x_val(i) -= dx;
      dy.col(i) = (yi - y0)/dx;
    }
    drake::math::initializeAutoDiffGivenGradientMatrix(y0, dy, *y);

    // Testing - looking at gradient values
    double max_element = dy(0, 0);
    double max_idx_i = 0;
    double max_idx_j = 0;
    for (int i = 0; i < dy.rows(); i++)
      for (int j = 0; j < dy.cols(); j++) {
        if (dy(i, j) > max_element) {
          max_element = dy(i, j);
          max_idx_i = i;
          max_idx_j = j;
        }
      }
    if (max_element > 1e3) {
      std::cout << this->get_description();
      std::cout << ":  gradient = " << max_element;
      std::cout << ",  max_idx_i = " << max_idx_i;
      std::cout << ",  max_idx_j = " << max_idx_j << std::endl;
    }
    if (this->get_description().compare("dynamics_constraint") == 0) {
      goldilocks_models::writeCSV("dyn_constraint_grad.csv", dy);
    }
    else if (this->get_description().compare("kinematics_constraint 24") == 0) {
      goldilocks_models::writeCSV("kin_constraint_grad.csv", dy);
    }
    else if (this->get_description().compare("impact_constraint") == 0) {
      goldilocks_models::writeCSV("impact_constraint_grad.csv", dy);
    }
    else if (this->get_description().compare("rom_dyn_constraint") == 0) {
      goldilocks_models::writeCSV("rom_dyn_constraint_grad.csv", dy);
    }


    // // central differencing
    // double dx = 1e-8;

    // VectorXd x_val = autoDiffToValueMatrix(x);
    // VectorXd y0,yi;
    // EvaluateConstraint(x_val,y0);

    // MatrixXd dy = MatrixXd(y0.size(),x_val.size());
    // for (int i=0; i < x_val.size(); i++) {
    //   x_val(i) -= dx/2;
    //   EvaluateConstraint(x_val,y0);
    //   x_val(i) += dx;
    //   EvaluateConstraint(x_val,yi);
    //   x_val(i) -= dx/2;
    //   dy.col(i) = (yi - y0)/dx;
    // }
    // EvaluateConstraint(x_val,y0);
    // initializeAutoDiffGivenGradientMatrix(y0, dy, y);
}



template <typename T>
DirconDynamicConstraint<T>::DirconDynamicConstraint(
    const MultibodyPlant<T>& plant, DirconKinematicDataSet<T>& constraints,
    bool is_floating_base,
    std::vector<double> var_scale) :
  DirconDynamicConstraint(plant, constraints, plant.num_positions(),
                          plant.num_velocities(), plant.num_actuators(),
                          constraints.countConstraints(),
                          (is_floating_base)? 1:0,
                          var_scale) {}

template <typename T>
DirconDynamicConstraint<T>::DirconDynamicConstraint(
    const MultibodyPlant<T>& plant, DirconKinematicDataSet<T>& constraints,
    int num_positions, int num_velocities, int num_inputs,
    int num_kinematic_constraints, int num_quat_slack,
    std::vector<double> var_scale)
    : DirconAbstractConstraint<T>(num_positions + num_velocities,
          1 + 2 *(num_positions+ num_velocities) + (2 * num_inputs) +
          (4 * num_kinematic_constraints) + num_quat_slack,
          Eigen::VectorXd::Zero(num_positions + num_velocities),
          Eigen::VectorXd::Zero(num_positions + num_velocities),
          "dynamics_constraint"),
      plant_(plant),
      constraints_(&constraints),
      num_states_{num_positions+num_velocities}, num_inputs_{num_inputs},
      num_kinematic_constraints_{num_kinematic_constraints},
      num_positions_{num_positions}, num_velocities_{num_velocities},
      num_quat_slack_{num_quat_slack},
      omega_scale_{var_scale[0]},
      input_scale_{var_scale[1]},
      force_scale_{var_scale[2]},
      time_scale_{var_scale[3]},
      quaternion_scale_{var_scale[4]} {}

// The format of the input to the eval() function is the
// tuple { timestep, state 0, state 1, input 0, input 1, force 0, force 1},
// which has a total length of 1 + 2*num_states + 2*num_inputs + dim*num_constraints.
template <typename T>
void DirconDynamicConstraint<T>::EvaluateConstraint(
    const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* y) const {
  DRAKE_ASSERT(x.size() == 1 + (2 * num_states_) + (2 * num_inputs_) +
      4*(num_kinematic_constraints_) + num_quat_slack_);

  double v_c_scale = 10;
  double gamma_scale = 0.002;

  // Extract our input variables:
  // h - current time (knot) value
  // x0, x1 state vector at time steps k, k+1
  // u0, u1 input vector at time steps k, k+1
  const T h = x(0) * time_scale_;
  // const VectorX<T> x0 = x.segment(1, num_states_);
  VectorX<T> x0(num_states_);
  x0 << x.segment(1, 4) * quaternion_scale_,
        x.segment(1 + 4, num_positions_ - 4),
        x.segment(1 + num_positions_, num_velocities_)*omega_scale_;
  // const VectorX<T> x1 = x.segment(1 + num_states_, num_states_);
  VectorX<T> x1(num_states_);
  x1 << x.segment(1 + num_states_, 4) * quaternion_scale_,
        x.segment(1 + num_states_ + 4, num_positions_ - 4),
        x.segment(1 + num_states_ + num_positions_, num_velocities_)*omega_scale_;
  const VectorX<T> u0 = x.segment(1 + (2 * num_states_), num_inputs_)*input_scale_;
  const VectorX<T> u1 = x.segment(1 + (2 * num_states_) + num_inputs_, num_inputs_)*input_scale_;
  const VectorX<T> l0 = x.segment(1 + 2 * (num_states_ + num_inputs_),
                            num_kinematic_constraints_)*force_scale_;
  const VectorX<T> l1 = x.segment(1 + 2 * (num_states_ + num_inputs_) +
      num_kinematic_constraints_, num_kinematic_constraints_)*force_scale_;
  const VectorX<T> lc = x.segment(1 + 2 * (num_states_ + num_inputs_) +
      2*num_kinematic_constraints_, num_kinematic_constraints_)*force_scale_;
  const VectorX<T> vc = x.segment(1 + 2 * (num_states_ + num_inputs_) +
      3*num_kinematic_constraints_, num_kinematic_constraints_) /**omega_scale_*/ * v_c_scale;
  const VectorX<T> gamma = x.tail(num_quat_slack_) /** gamma_scale*/;

  auto context0 = multibody::createContext(plant_, x0, u0);
  constraints_->updateData(*context0, l0);
  const VectorX<T> xdot0 = constraints_->getXDot();

  auto context1 = multibody::createContext(plant_, x1, u1);
  constraints_->updateData(*context1, l1);
  const VectorX<T> xdot1 = constraints_->getXDot();

  // Cubic interpolation to get xcol and xdotcol.
  const VectorX<T> xcol = 0.5 * (x0 + x1) + h / 8 * (xdot0 - xdot1);
  const VectorX<T> xdotcol = -1.5 * (x0 - x1) / h - .25 * (xdot0 + xdot1);
  const VectorX<T> ucol = 0.5 * (u0 + u1);

  auto contextcol = multibody::createContext(plant_, xcol, ucol);
  constraints_->updateData(*contextcol, lc);
  auto g = constraints_->getXDot();
  VectorX<T> vc_in_qdot_space(num_positions_);
  plant_.MapVelocityToQDot(*contextcol,
      constraints_->getJ().transpose()*vc, &vc_in_qdot_space);
  g.head(num_positions_) += vc_in_qdot_space;
  if (num_quat_slack_ > 0) {
    // Assume the floating base coordinates is in the first four elements.
    g.head(4) += xcol.head(4) * gamma;
  }

  double vel_scale = 60;
  double accel_scale = 20 * vel_scale;
  double toe_scale = 50;
  VectorX<T> output = xdotcol - g;
  output.head(num_positions_) /= vel_scale;
  output.tail(num_velocities_) /= accel_scale;
  // output.tail(2) /= toe_scale;
  *y = output;
}

template <typename T>
Binding<Constraint> AddDirconConstraint(
    std::shared_ptr<DirconDynamicConstraint<T>> constraint,
    const Eigen::Ref<const VectorXDecisionVariable>& timestep,
    const Eigen::Ref<const VectorXDecisionVariable>& state,
    const Eigen::Ref<const VectorXDecisionVariable>& next_state,
    const Eigen::Ref<const VectorXDecisionVariable>& input,
    const Eigen::Ref<const VectorXDecisionVariable>& next_input,
    const Eigen::Ref<const VectorXDecisionVariable>& force,
    const Eigen::Ref<const VectorXDecisionVariable>& next_force,
    const Eigen::Ref<const VectorXDecisionVariable>& collocation_force,
    const Eigen::Ref<const VectorXDecisionVariable>& collocation_position_slack,
    MathematicalProgram* prog) {
  DRAKE_DEMAND(timestep.size() == 1);
  DRAKE_DEMAND(state.size() == constraint->num_states());
  DRAKE_DEMAND(next_state.size() == constraint->num_states());
  DRAKE_DEMAND(input.size() == constraint->num_inputs());
  DRAKE_DEMAND(next_input.size() == constraint->num_inputs());
  DRAKE_DEMAND(force.size() == constraint->num_kinematic_constraints());
  DRAKE_DEMAND(next_force.size() == constraint->num_kinematic_constraints());
  DRAKE_DEMAND(collocation_force.size() == constraint->num_kinematic_constraints());
  DRAKE_DEMAND(collocation_position_slack.size() ==
      constraint->num_kinematic_constraints());
  return prog->AddConstraint(constraint, {timestep, state, next_state, input,
                                          next_input, force, next_force,
                                          collocation_force,
                                          collocation_position_slack});
}

template <typename T>
DirconKinematicConstraint<T>::DirconKinematicConstraint(
    const MultibodyPlant<T>& plant, DirconKinematicDataSet<T>& constraints,
    DirconKinConstraintType type,
    std::vector<double> var_scale) :
    DirconKinematicConstraint(plant, constraints,
                            std::vector<bool>(constraints.countConstraints(),
                            false),
                            Eigen::VectorXd::Zero(constraints.countConstraints()),
                            type, plant.num_positions(),
                            plant.num_velocities(), plant.num_actuators(),
                            constraints.countConstraints(),
                            var_scale) {}

template <typename T>
DirconKinematicConstraint<T>::DirconKinematicConstraint(
    const MultibodyPlant<T>& plant, DirconKinematicDataSet<T>& constraints,
    std::vector<bool> is_constraint_relative, drake::VectorX<double> phi_vals,
    DirconKinConstraintType type,
    std::vector<double> var_scale) :
    DirconKinematicConstraint(plant, constraints, is_constraint_relative,
                              phi_vals, type,
                              plant.num_positions(),
                              plant.num_velocities(),
                              plant.num_actuators(),
                              constraints.countConstraints(),
                              var_scale) {}

template <typename T>
DirconKinematicConstraint<T>::DirconKinematicConstraint(
    const MultibodyPlant<T>& plant, DirconKinematicDataSet<T>& constraints,
    std::vector<bool> is_constraint_relative, drake::VectorX<double> phi_vals,
    DirconKinConstraintType type,
    int num_positions, int num_velocities, int num_inputs,
    int num_kinematic_constraints,
    std::vector<double> var_scale) :
    DirconAbstractConstraint<T>(type*num_kinematic_constraints, num_positions +
        num_velocities + num_inputs + num_kinematic_constraints +
        std::count(is_constraint_relative.begin(),
                   is_constraint_relative.end(), true),
        VectorXd::Zero(type*num_kinematic_constraints),
        VectorXd::Zero(type*num_kinematic_constraints),
        "kinematics_constraint "+std::to_string(type*num_kinematic_constraints)),
      plant_(plant),
      constraints_(&constraints),
      num_states_{num_positions+num_velocities}, num_inputs_{num_inputs},
      num_kinematic_constraints_{num_kinematic_constraints},
      num_positions_{num_positions}, num_velocities_{num_velocities},
      type_{type}, is_constraint_relative_{is_constraint_relative},
      phi_vals_{phi_vals},
      n_relative_{static_cast<int>(std::count(is_constraint_relative.begin(),
      is_constraint_relative.end(), true))},
      omega_scale_{var_scale[0]},
      input_scale_{var_scale[1]},
      force_scale_{var_scale[2]},
      quaternion_scale_{var_scale[4]} {
  relative_map_ = MatrixXd::Zero(num_kinematic_constraints_, n_relative_);
  int j = 0;
  for (int i=0; i < num_kinematic_constraints_; i++) {
    if (is_constraint_relative_[i]) {
      relative_map_(i, j) = 1;
      j++;
    }
  }
}

template <typename T>
void DirconKinematicConstraint<T>::EvaluateConstraint(
    const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* y) const {
  DRAKE_ASSERT(x.size() == num_states_ + num_inputs_ +
                           num_kinematic_constraints_ + n_relative_);

  // Extract our input variables:
  // x0, x1 state vector at time steps k, k+1
  // u0, u1 input vector at time steps k, k+1
  VectorX<T> state(num_states_);
  state << x.segment(0, 4) * quaternion_scale_,
        x.segment(4, num_positions_ - 4),
        x.segment(num_positions_, num_velocities_)*omega_scale_;
  const VectorX<T> input = x.segment(num_states_, num_inputs_)*input_scale_;
  const VectorX<T> force = x.segment(num_states_ + num_inputs_,
                               num_kinematic_constraints_)*force_scale_;
  const VectorX<T> offset = x.segment(num_states_ + num_inputs_ +
                                num_kinematic_constraints_, n_relative_);
  auto context = multibody::createContext(plant_, state, input);
  constraints_->updateData(*context, force);

  double vel_scale = 10;
  double accel_scale = 300 * 20;  //300 * 4
  // An even better scaling is to scale the distance constraint (four-bar linkage).

  switch (type_) {
    case kAll:
      *y = VectorX<T>(3*num_kinematic_constraints_);
      *y << constraints_->getC() + relative_map_*offset - phi_vals_,
            constraints_->getCDot() / vel_scale,
            constraints_->getCDDot() / accel_scale;
      break;
    case kAccelAndVel:
      *y = VectorX<T>(2*num_kinematic_constraints_);
      *y << constraints_->getCDot() / vel_scale,
            constraints_->getCDDot() / accel_scale;
      break;
    case kAccelOnly:
      *y = VectorX<T>(1*num_kinematic_constraints_);
      *y << constraints_->getCDDot() / accel_scale;
      break;
  }
}

template <typename T>
DirconImpactConstraint<T>::DirconImpactConstraint(
    const MultibodyPlant<T>& plant, DirconKinematicDataSet<T>& constraints,
    std::vector<double> var_scale) :
  DirconImpactConstraint(plant, constraints, plant.num_positions(),
                         plant.num_velocities(),
                         constraints.countConstraints(),
                         var_scale) {}

template <typename T>
DirconImpactConstraint<T>::DirconImpactConstraint(
    const MultibodyPlant<T>& plant, DirconKinematicDataSet<T>& constraints,
    int num_positions, int num_velocities, int num_kinematic_constraints,
    std::vector<double> var_scale) :
        DirconAbstractConstraint<T>(num_velocities, num_positions +
            2*num_velocities + num_kinematic_constraints,
            VectorXd::Zero(num_velocities), VectorXd::Zero(num_velocities),
            "impact_constraint"),
        plant_(plant),
        constraints_(&constraints),
        num_states_{num_positions+num_velocities},
        num_kinematic_constraints_{num_kinematic_constraints},
        num_positions_{num_positions}, num_velocities_{num_velocities},
        omega_scale_{var_scale[0]},
        input_scale_{var_scale[1]},
        force_scale_{var_scale[2]},
        quaternion_scale_{var_scale[4]} {}


// The format of the input to the eval() function is the
// tuple { state 0, impulse, velocity 1},
template <typename T>
void DirconImpactConstraint<T>::EvaluateConstraint(
    const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* y) const {
  DRAKE_ASSERT(x.size() == 2 * num_velocities_ + num_positions_ +
                           num_kinematic_constraints_);

  // Extract our input variables:
  // x0, state vector at time k^-
  // impulse, impulsive force at impact
  // v1, post-impact velocity at time k^+
  VectorX<T> x0(num_states_);
  x0 << x.segment(0, 4) * quaternion_scale_,
        x.segment(4, num_positions_ - 4),
        x.segment(num_positions_, num_velocities_)*omega_scale_;
  const VectorX<T> impulse = x.segment(num_states_, num_kinematic_constraints_)/**force_scale_*/;
  const VectorX<T> v1 = x.segment(num_states_ + num_kinematic_constraints_,
                            num_velocities_)*omega_scale_;

  const VectorX<T> v0 = x0.tail(num_velocities_);

  // vp = vm + M^{-1}*J^T*Lambda
  const VectorX<T> u = VectorXd::Zero(plant_.num_actuators()).template cast<T>();

  auto context = multibody::createContext(plant_, x0, u);

  constraints_->updateData(*context, impulse);

  MatrixX<T> M(num_velocities_, num_velocities_);
  plant_.CalcMassMatrixViaInverseDynamics(*context, &M);

  double accel_scale = 12;
  double xyz_scale = 50;
  VectorX<T> output = M*(v1 - v0) - constraints_->getJ().transpose()*impulse;
  output /= accel_scale;
  output.segment(3,3) /= xyz_scale;
  *y = output;
}

// Explicitly instantiates on the most common scalar types.
template class DirconAbstractConstraint<double>;
template class DirconDynamicConstraint<double>;
template class DirconDynamicConstraint<AutoDiffXd>;
template class DirconKinematicConstraint<double>;
template class DirconKinematicConstraint<AutoDiffXd>;
template class DirconImpactConstraint<double>;
template class DirconImpactConstraint<AutoDiffXd>;


}  // namespace trajectory_optimization
}  // namespace systems
}  // namespace dairlib

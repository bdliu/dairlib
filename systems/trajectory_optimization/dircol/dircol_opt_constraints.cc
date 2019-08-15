#include "dircol_opt_constraints.h"
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "multibody/multibody_utils.h"

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
DircolAbstractConstraint<T>::DircolAbstractConstraint(int num_constraints, int num_vars,
                                                      const VectorXd& lb,
                                                      const VectorXd& ub,
                                                      const std::string& description)
  : Constraint(num_constraints, num_vars, lb, ub, description) {
}

template <>
void DircolAbstractConstraint<double>::DoEval(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    Eigen::VectorXd* y) const {
  EvaluateConstraint(x, y);
}

template <>
void DircolAbstractConstraint<AutoDiffXd>::DoEval(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    Eigen::VectorXd* y) const {
  AutoDiffVecXd y_t;
  EvaluateConstraint(initializeAutoDiff(x), &y_t);
  *y = autoDiffToValueMatrix(y_t);
}

template <typename T>
void DircolAbstractConstraint<T>::DoEval(
    const Eigen::Ref<const VectorX<drake::symbolic::Variable>>& x,
    VectorX<drake::symbolic::Expression>* y) const {
  throw std::logic_error(
      "DircolAbstractConstraint does not support symbolic evaluation.");
}

template <>
void DircolAbstractConstraint<AutoDiffXd>::DoEval(
    const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd* y) const {
  EvaluateConstraint(x,y);
}

template <>
void DircolAbstractConstraint<double>::DoEval(
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
CubicInterpolationConstraint<T>::CubicInterpolationConstraint(
    const MultibodyPlant<T>& plant, DircolKinematicDataSet<T>& constraints) :
  CubicInterpolationConstraint(plant, constraints,
      plant.num_positions(), plant.num_velocities(), plant.num_actuators()) {}

template <typename T>
CubicInterpolationConstraint<T>::CubicInterpolationConstraint(
    const MultibodyPlant<T>& plant, DircolKinematicDataSet<T>& constraints,
    int num_positions, int num_velocities, int num_inputs)
    : DircolAbstractConstraint<T>(
          2 * (num_positions + num_velocities) + num_inputs,
          1 + 3 * (num_positions + num_velocities + num_inputs),
          Eigen::VectorXd::Zero(num_positions + num_velocities + num_inputs),
          Eigen::VectorXd::Zero(num_positions + num_velocities + num_inputs)),
      plant_(plant),
      constraints_(&constraints),
      num_states_{num_positions+num_velocities}, num_inputs_{num_inputs},
      num_kinematic_constraints_{constraints.countConstraints()},
      num_positions_{num_positions}, num_velocities_{num_velocities},
      num_xv_{num_positions + num_velocities + num_inputs},
      num_xvu_{num_positions + 2*num_velocities + num_inputs} {}

template <typename T>
void CubicInterpolationConstraint<T>::EvaluateConstraint(
    const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* y) const {
  DRAKE_ASSERT(x.size() == 1 + 3 * num_xvu_);

  // Extract our input variables:
  // h - current time (knot) value
  // x0, state vector at time steps k
  // u0, input vector at time steps k
  // x1, state vector at time steps k+1
  // u1, input vector at time steps k+1
  // xc, state vector at the collocation point between time steps k and k+1
  // uc, input vector at the collocation point between time steps k and k+1
  const T h = x(0);
  const VectorX<T> x0 = x.segment(1, num_states_);
  const VectorX<T> dv0 = x.segment(1 + num_states_, num_velocities_);
  const VectorX<T> u0 = x.segment(1 + num_xv_, num_inputs_);
  const VectorX<T> x1 = x.segment(1 + num_xvu_, num_states_);
  const VectorX<T> dv1 = x.segment(1 + num_xvu_ + num_states_, num_velocities_);
  const VectorX<T> u1 = x.segment(1 + num_xvu_ + num_xv_, num_inputs_);
  const VectorX<T> xc = x.segment(1 + 2 * num_xvu_, num_states_);
  const VectorX<T> dvc = x.segment(1 + 2 * num_xvu_ + num_states_, num_velocities_);
  const VectorX<T> uc = x.segment(1 + 2 * num_xvu_ + num_xv_, num_inputs_);

  // Get xdot0, xdot1, xdotc
  const VectorX<T> lambda = VectorXd::Zero(num_kinematic_constraints_).template cast<T>();
  auto context0 = multibody::createContext(plant_, x0, u0);
  auto context1 = multibody::createContext(plant_, x1, u1);
  auto contextc = multibody::createContext(plant_, xc, uc);
  constraints_->updateData(*context0, lambda);
  constraints_->updateData(*context1, lambda);
  constraints_->updateData(*contextc, lambda);
  const VectorX<T> qdot0 = constraints_->getQDot();
  const VectorX<T> qdot1 = constraints_->getQDot();
  const VectorX<T> qdotc = constraints_->getQDot();
  VectorX<T> xdot0(num_states_);
  VectorX<T> xdot1(num_states_);
  VectorX<T> xdotc(num_states_);
  xdot0 << qdot0, dv0;
  xdot1 << qdot1, dv1;
  xdotc << qdotc, dvc;

  // Cubic interpolation to get xcol and xdotcol.
  const VectorX<T> xc_interp = 0.5 * (x0 + x1) + h / 8 * (xdot0 - xdot1);
  const VectorX<T> xdotc_interp = -1.5 * (x0 - x1) / h - .25 * (xdot0 + xdot1);
  const VectorX<T> uc_interp = 0.5 * (u0 + u1);
  VectorX<T> xc_xdotc_uc_interp(2 * num_states_ + num_inputs_);
  xc_xdotc_uc_interp << xc_interp, xdotc_interp, uc_interp;

  // Compose xdotc from x
  VectorX<T> xc_xdotc_uc(2 * num_states_ + num_inputs_);
  xc_xdotc_uc << xc, xdotc, uc;

  *y = xc_xdotc_uc_interp - xc_xdotc_uc;
}



template <typename T>
DircolDynamicConstraint<T>::DircolDynamicConstraint(
    const MultibodyPlant<T>& plant, DircolKinematicDataSet<T>& constraints,
    bool is_floating_base) :
  DircolDynamicConstraint(plant, constraints, plant.num_positions(),
                          plant.num_velocities(), plant.num_actuators(),
                          constraints.countConstraints(),
                          (is_floating_base)? 1:0) {}

template <typename T>
DircolDynamicConstraint<T>::DircolDynamicConstraint(
    const MultibodyPlant<T>& plant, DircolKinematicDataSet<T>& constraints,
    int num_positions, int num_velocities, int num_inputs,
    int num_kinematic_constraints, int num_quat_slack)
    : DircolAbstractConstraint<T>(num_positions + num_velocities,
          num_positions + 2 * num_velocities + num_inputs + num_quat_slack,
          Eigen::VectorXd::Zero(num_positions + num_velocities),
          Eigen::VectorXd::Zero(num_positions + num_velocities)),
      plant_(plant),
      constraints_(&constraints),
      num_states_{num_positions+num_velocities}, num_inputs_{num_inputs},
      num_kinematic_constraints_{num_kinematic_constraints},
      num_positions_{num_positions}, num_velocities_{num_velocities},
      num_quat_slack_{num_quat_slack} {}

// The format of the input to the eval() function is the
// tuple { state, dv, input, force, slack variable},
template <typename T>
void DircolDynamicConstraint<T>::EvaluateConstraint(
    const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* y) const {
  DRAKE_ASSERT(x.size() == num_states_ + num_inputs_ + num_velocities_ +
      num_quat_slack_);

  // Extract our input variables:
  // state, state vector at time steps k
  // dv, acceleration of generalized position vector at time steps k
  // u, input vector at time steps k
  // lambda, constraint force vector at time steps k
  // gamma, slack variable for unit quaternion norm
  const VectorX<T> state = x.segment(0, num_states_);
  const VectorX<T> dv = x.segment(num_states_, num_velocities_);
  const VectorX<T> u = x.segment(num_states_ + num_velocities_, num_inputs_);
  const VectorX<T> lambda = x.segment(num_states_ + num_velocities_ + num_inputs_,
                            num_kinematic_constraints_);
  const VectorX<T> gamma = x.tail(num_quat_slack_);

  auto context = multibody::createContext(plant_, state, u);
  constraints_->updateData(*context, lambda);

  auto qdot = constraints_->getQDot();
  if (num_quat_slack_ > 0) {
    // Assume the floating base coordinates is in the first four elements.
    qdot.head(4) += state.head(4) * gamma;
  }
  auto dq_residual = state.head(num_positions_) - qdot;

  // right_hand_side is the right hand side of the system's equations:
  // M*vdot -J^T*f = right_hand_side.
  auto J = constraints_->getJ();
  auto M = constraints_->getM();
  auto right_hand_side = constraints_->getRightHandSide();
  auto dv_residual = M*dv - J.transpose()*lambda - right_hand_side;

  VectorX<T> dx_residual(num_states_);
  dx_residual << dq_residual, dv_residual;
  *y = dx_residual;
}


template <typename T>
DircolKinematicConstraint<T>::DircolKinematicConstraint(
    const MultibodyPlant<T>& plant, DircolKinematicDataSet<T>& constraints,
    DircolKinConstraintType type) :
    DircolKinematicConstraint(plant, constraints,
                            std::vector<bool>(constraints.countConstraints(),false),
                            Eigen::VectorXd::Zero(constraints.countConstraints()),
                            type,
                            plant.num_positions(),
                            plant.num_velocities(),
                            plant.num_actuators(),
                            constraints.countConstraints()) {}

template <typename T>
DircolKinematicConstraint<T>::DircolKinematicConstraint(
    const MultibodyPlant<T>& plant, DircolKinematicDataSet<T>& constraints,
    std::vector<bool> is_constraint_relative, drake::VectorX<double> phi_vals,
    DircolKinConstraintType type) :
    DircolKinematicConstraint(plant, constraints,
                              is_constraint_relative,
                              phi_vals,
                              type,
                              plant.num_positions(),
                              plant.num_velocities(),
                              plant.num_actuators(),
                              constraints.countConstraints()) {}

template <typename T>
DircolKinematicConstraint<T>::DircolKinematicConstraint(
    const MultibodyPlant<T>& plant, DircolKinematicDataSet<T>& constraints,
    std::vector<bool> is_constraint_relative,
    drake::VectorX<double> phi_vals,
    DircolKinConstraintType type,
    int num_positions, int num_velocities, int num_inputs,
    int num_kinematic_constraints) :
    DircolAbstractConstraint<T>(type*num_kinematic_constraints,
        num_positions + num_velocities + num_inputs +
            num_kinematic_constraints +
            std::count(is_constraint_relative.begin(),
                       is_constraint_relative.end(), true),
        VectorXd::Zero(type*num_kinematic_constraints),
        VectorXd::Zero(type*num_kinematic_constraints)),
      plant_(plant),
      constraints_(&constraints),
      num_states_{num_positions+num_velocities}, num_inputs_{num_inputs},
      num_kinematic_constraints_{num_kinematic_constraints},
      num_positions_{num_positions}, num_velocities_{num_velocities},
      type_{type}, is_constraint_relative_{is_constraint_relative},
      phi_vals_{phi_vals},
      n_relative_{static_cast<int>(std::count(is_constraint_relative.begin(),
      is_constraint_relative.end(), true))} {
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
void DircolKinematicConstraint<T>::EvaluateConstraint(
    const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* y) const {
  DRAKE_ASSERT(x.size() == num_states_ + num_inputs_ +
                           num_kinematic_constraints_ + n_relative_);

  // Extract our input variables:
  // x, state vector at time steps k
  // u, input vector at time steps k
  // lambda, force vector at time steps k
  // offset, position offset vector at time steps k
  const VectorX<T> state = x.segment(0, num_states_);
  const VectorX<T> input = x.segment(num_states_, num_inputs_);
  const VectorX<T> force = x.segment(num_states_ + num_inputs_,
                               num_kinematic_constraints_);
  const VectorX<T> offset = x.segment(num_states_ + num_inputs_ +
                                num_kinematic_constraints_, n_relative_);
  auto context = multibody::createContext(plant_, state, input);
  constraints_->updateData(*context, force);
  switch (type_) {
    case kPosOnly:
      *y = constraints_->getC() + relative_map_*offset - phi_vals_;
      break;
  }
}

template <typename T>
DircolImpactConstraint<T>::DircolImpactConstraint(
    const MultibodyPlant<T>& plant, DircolKinematicDataSet<T>& constraints) :
  DircolImpactConstraint(plant, constraints, plant.num_positions(),
                         plant.num_velocities(),
                         constraints.countConstraints()) {}

template <typename T>
DircolImpactConstraint<T>::DircolImpactConstraint(
    const MultibodyPlant<T>& plant, DircolKinematicDataSet<T>& constraints,
    int num_positions, int num_velocities, int num_kinematic_constraints) :
    DircolAbstractConstraint<T>(num_kinematic_constraints + num_velocities,
        num_positions + 2*num_velocities + num_kinematic_constraints,
        VectorXd::Zero(num_kinematic_constraints + num_velocities),
        VectorXd::Zero(num_kinematic_constraints + num_velocities)),
    plant_(plant),
    constraints_(&constraints),
    num_states_{num_positions+num_velocities},
    num_kinematic_constraints_{num_kinematic_constraints},
    num_positions_{num_positions}, num_velocities_{num_velocities} {}


// The format of the input to the eval() function is the
// tuple { state 0, impulse, velocity 1},
template <typename T>
void DircolImpactConstraint<T>::EvaluateConstraint(
    const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* y) const {
  DRAKE_ASSERT(x.size() == 2 * num_velocities_ + num_positions_ +
                           num_kinematic_constraints_);

  // Extract our input variables:
  // x0, state vector at time k^-
  // impulse, impulsive force at impact
  // v1, post-impact velocity at time k^+
  const VectorX<T> x0 = x.segment(0, num_states_);
  const VectorX<T> impulse = x.segment(num_states_, num_kinematic_constraints_);
  const VectorX<T> v1 = x.segment(num_states_ + num_kinematic_constraints_,
                            num_velocities_);

  const VectorX<T> v0 = x0.tail(num_velocities_);

  // vp = vm + M^{-1}*J^T*Lambda
  const VectorX<T> u = VectorXd::Zero(plant_.num_actuators()).template cast<T>();

  auto context = multibody::createContext(plant_, x0, u);
  constraints_->updateData(*context, impulse);

  // constraint C17 and C18 in Ayonga's paper
  VectorX<T> cdotpost_vresidual(num_kinematic_constraints_ + num_velocities_);
  cdotpost_vresidual << constraints_->getJ() * v1,
      constraints_->getM()*(v1 - v0) - constraints_->getJ().transpose()*impulse;

  *y = cdotpost_vresidual;
}


template <typename T>
Binding<Constraint> AddDircolConstraint(
    std::shared_ptr<DircolDynamicConstraint<T>> constraint,
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


// Explicitly instantiates on the most common scalar types.
template class DircolAbstractConstraint<double>;
template class CubicInterpolationConstraint<double>;
template class CubicInterpolationConstraint<AutoDiffXd>;
template class DircolDynamicConstraint<double>;
template class DircolDynamicConstraint<AutoDiffXd>;
template class DircolKinematicConstraint<double>;
template class DircolKinematicConstraint<AutoDiffXd>;
template class DircolImpactConstraint<double>;
template class DircolImpactConstraint<AutoDiffXd>;


}  // namespace trajectory_optimization
}  // namespace systems
}  // namespace dairlib

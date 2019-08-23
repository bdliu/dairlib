#include <chrono>

#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "systems/trajectory_optimization/dircon_kinematic_data_set.h"
#include "multibody/multibody_utils.h"

namespace dairlib {

using std::vector;
using drake::VectorX;
using drake::MatrixX;
using drake::AutoDiffXd;
using drake::multibody::MultibodyPlant;
using drake::systems::Context;
using drake::math::DiscardGradient;

template <typename T>
DirconKinematicDataSet<T>::DirconKinematicDataSet(
    const MultibodyPlant<T>& plant,
    vector<DirconKinematicData<T>*>* constraints) :
    plant_(plant),
    constraints_(constraints),
    num_positions_(plant.num_positions()),
    num_velocities_(plant.num_velocities()),
    cache_(500) {
  // Initialize matrices
  constraint_count_ = 0;
  for (uint i=0; i < constraints_->size(); i++) {
    constraint_count_ += (*constraints_)[i]->getLength();
  }
  c_ = VectorX<T>(constraint_count_);
  cdot_ = VectorX<T>(constraint_count_);
  J_ = MatrixX<T>(constraint_count_, num_velocities_);
  Jdotv_ = VectorX<T>(constraint_count_);
  cddot_ = VectorX<T>(constraint_count_);
  vdot_ = VectorX<T>(num_velocities_);
  xdot_ = VectorX<T>(num_positions_ + num_velocities_);
  M_ = MatrixX<T>(num_velocities_, num_velocities_);
  right_hand_side_ = VectorX<T>(num_velocities_);

  // testing
  row_idx_stack_set_to_0_.clear();
  for (unsigned int i = 0; i < constraints->size(); i++) {
    row_idx_stack_set_to_0_.insert(row_idx_stack_set_to_0_.end(),
                       constraints->at(i)->row_idx_set_to_0_.begin(),
                       constraints->at(i)->row_idx_set_to_0_.end() );
  }
}


template <typename T>
void DirconKinematicDataSet<T>::updateData(const Context<T>& context,
                                           const VectorX<T>& forces) {
  const auto state = plant_.GetPositionsAndVelocities(context);

  const VectorX<T> q = state.head(num_positions_);
  const VectorX<T> v = state.tail(num_velocities_);

  VectorX<T> input = multibody::getInput(plant_, context);

  // Create a CacheKey element by discarding gradient information (if AutoDiff)
  CacheKey key{DiscardGradient(state), DiscardGradient(forces),
      DiscardGradient(input)};

  if (cache_.Contains(key)) {
    auto data = cache_.GetData(key);
    c_ = data.c_;
    cdot_ = data.cdot_;
    J_ = data.J_;
    Jdotv_ = data.Jdotv_;
    cddot_ = data.cddot_;
    vdot_ = data.vdot_;
    xdot_ = data.xdot_;
  } else {
    int index = 0;
    int n;
    for (uint i=0; i < constraints_->size(); i++) {
      (*constraints_)[i]->updateConstraint(context);

      n = (*constraints_)[i]->getLength();
      c_.segment(index, n) = (*constraints_)[i]->getC();
      cdot_.segment(index, n) = (*constraints_)[i]->getCDot();
      J_.block(index, 0, n, num_velocities_) = (*constraints_)[i]->getJ();
      Jdotv_.segment(index, n) = (*constraints_)[i]->getJdotv();

      index += n;
    }

    plant_.CalcMassMatrixViaInverseDynamics(context, &M_);

    // right_hand_side is the right hand side of the system's equations:
    // M*vdot -J^T*f = right_hand_side.
    // BiasTerm is C(q,v) in manipulator equations
    plant_.CalcBiasTerm(context, &right_hand_side_);

    right_hand_side_ = -right_hand_side_ +
        plant_.MakeActuationMatrix() * input +
        plant_.CalcGravityGeneralizedForces(context) +
        getJ().transpose() * forces;

    vdot_ = M_.llt().solve(right_hand_side_);
    // std::cout << "residual = " << (M_*vdot_ - right_hand_side_).norm() << std::endl;

    cddot_ = Jdotv_ + J_*vdot_;

    VectorX<T> q_dot(num_positions_);
    plant_.MapVelocityToQDot(context, v, &q_dot);
    xdot_ << q_dot, vdot_;

    // testing
    for (unsigned int i = 0; i < row_idx_stack_set_to_0_.size(); i++) {
      if (row_idx_stack_set_to_0_[i]) {
        c_[i] = 0;
        cdot_[i] = 0;
        cddot_[i] = 0;
      }
    }
    // std::cout << "cddot_= " << cddot_.transpose() << std::endl;

    CacheData data{c_, cdot_, J_, Jdotv_, cddot_, vdot_, xdot_};

    cache_.AddData(key, data);
  }
}

template <typename T>
int DirconKinematicDataSet<T>::countConstraints() {
  return constraint_count_;
}

template <typename T>
int DirconKinematicDataSet<T>::getNumConstraintObjects() {
  return constraints_->size();
}

template <typename T>
VectorX<T> DirconKinematicDataSet<T>::getC() {
  return c_;
}

template <typename T>
VectorX<T> DirconKinematicDataSet<T>::getCDot() {
  return cdot_;
}

template <typename T>
MatrixX<T> DirconKinematicDataSet<T>::getJ() {
  return J_;
}

template <typename T>
VectorX<T> DirconKinematicDataSet<T>::getJdotv() {
  return Jdotv_;
}

template <typename T>
VectorX<T> DirconKinematicDataSet<T>::getCDDot() {
  return cddot_;
}

template <typename T>
VectorX<T> DirconKinematicDataSet<T>::getVDot() {
  return vdot_;
}

template <typename T>
VectorX<T> DirconKinematicDataSet<T>::getXDot() {
  return xdot_;
}

template <typename T>
DirconKinematicData<T>* DirconKinematicDataSet<T>::getConstraint(int index) {
  return (*constraints_)[index];
}


// Explicitly instantiates on the most common scalar types.
template class DirconKinematicDataSet<double>;
template class DirconKinematicDataSet<AutoDiffXd>;

}  // namespace dairlib

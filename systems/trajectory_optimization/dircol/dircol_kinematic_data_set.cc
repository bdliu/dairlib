#include <chrono>

#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "systems/trajectory_optimization/dircol/dircol_kinematic_data_set.h"
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
DircolKinematicDataSet<T>::DircolKinematicDataSet(
    const MultibodyPlant<T>& plant,
    vector<DircolKinematicData<T>*>* constraints) :
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
  // cddot_ = VectorX<T>(constraint_count_);
  qdot_ = VectorX<T>(num_positions_);
  // vdot_ = VectorX<T>(num_velocities_);
  // xdot_ = VectorX<T>(num_positions_ + num_velocities_);
  M_ = MatrixX<T>(num_velocities_, num_velocities_);
  right_hand_side_ = VectorX<T>(num_velocities_);
}


template <typename T>
void DircolKinematicDataSet<T>::updateData(const Context<T>& context,
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
    // cddot_ = data.cddot_;
    qdot_ = data.qdot_;
    // vdot_ = data.vdot_;
    // xdot_ = data.xdot_;
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

    // vdot_ = M_.llt().solve(right_hand_side_);
    // cddot_ = Jdotv_ + J_*vdot_;

    plant_.MapVelocityToQDot(context, v, &qdot_);
    // xdot_ << qdot_, vdot_;

    CacheData data{c_, cdot_, J_, Jdotv_, qdot_, M_, right_hand_side_};

    cache_.AddData(key, data);
  }
}

template <typename T>
int DircolKinematicDataSet<T>::countConstraints() {
  return constraint_count_;
}

template <typename T>
int DircolKinematicDataSet<T>::getNumConstraintObjects() {
  return constraints_->size();
}

template <typename T>
VectorX<T> DircolKinematicDataSet<T>::getC() {
  return c_;
}

template <typename T>
VectorX<T> DircolKinematicDataSet<T>::getCDot() {
  return cdot_;
}

template <typename T>
MatrixX<T> DircolKinematicDataSet<T>::getJ() {
  return J_;
}

template <typename T>
VectorX<T> DircolKinematicDataSet<T>::getJdotv() {
  return Jdotv_;
}

// template <typename T>
// VectorX<T> DircolKinematicDataSet<T>::getCDDot() {
//   return cddot_;
// }

template <typename T>
VectorX<T> DircolKinematicDataSet<T>::getQDot() {
  return qdot_;
}

// template <typename T>
// VectorX<T> DircolKinematicDataSet<T>::getVDot() {
//   return vdot_;
// }

// template <typename T>
// VectorX<T> DircolKinematicDataSet<T>::getXDot() {
//   return xdot_;
// }

template <typename T>
MatrixX<T> DircolKinematicDataSet<T>::getM() {
  return M_;
}

template <typename T>
VectorX<T> DircolKinematicDataSet<T>::getRightHandSide() {
  return right_hand_side_;
}

template <typename T>
DircolKinematicData<T>* DircolKinematicDataSet<T>::getConstraint(int index) {
  return (*constraints_)[index];
}


// Explicitly instantiates on the most common scalar types.
template class DircolKinematicDataSet<double>;
template class DircolKinematicDataSet<AutoDiffXd>;

}  // namespace dairlib
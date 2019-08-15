#include "dircol_kinematic_data.h"

namespace dairlib {

using std::shared_ptr;
using std::vector;
using drake::VectorX;
using drake::MatrixX;
using drake::solvers::Constraint;
using drake::multibody::MultibodyPlant;

template <typename T>
DircolKinematicData<T>::DircolKinematicData(const MultibodyPlant<T>& plant,
                                            int length) :
    plant_(plant),
    length_(length) {
  force_constraints_ = vector<shared_ptr<Constraint>>(0);
  c_ = VectorX<T>::Zero(length);
  cdot_ = VectorX<T>::Zero(length);
  J_ = MatrixX<T>::Zero(length, plant.num_velocities());
  Jdotv_ = VectorX<T>::Zero(length);
}

template <typename T>
DircolKinematicData<T>::~DircolKinematicData() {}

template <typename T>
VectorX<T> DircolKinematicData<T>::getC() {
  return c_;
}

template <typename T>
VectorX<T> DircolKinematicData<T>::getCDot() {
  return cdot_;
}

template <typename T>
MatrixX<T> DircolKinematicData<T>::getJ() {
  return J_;
}

template <typename T>
VectorX<T> DircolKinematicData<T>::getJdotv() {
  return Jdotv_;
}

template <typename T>
int DircolKinematicData<T>::getLength() {
  return length_;
}

template <typename T>
int DircolKinematicData<T>::numForceConstraints() {
  return force_constraints_.size();
}

template <typename T>
shared_ptr<Constraint> DircolKinematicData<T>::getForceConstraint(int index) {
  return force_constraints_[index];
}

// Explicitly instantiates on the most common scalar types.
template class DircolKinematicData<double>;
template class DircolKinematicData<drake::AutoDiffXd>;

}  // namespace dairlib


#include "systems/trajectory_optimization/dircol/dircol_options.h"

using std::vector;

namespace dairlib {
namespace systems {
namespace trajectory_optimization {

DircolOptions::DircolOptions(int n_constraints) {
  n_constraints_ = n_constraints;
  is_constraints_relative_ = vector<bool>(n_constraints_);
  for (int i=0; i < n_constraints_; i++) {
    is_constraints_relative_[i] = false;
  }
  phi_vals_ = Eigen::VectorXd::Zero(n_constraints_);
  start_constraint_type_ = DircolKinConstraintType::kPosOnly;
  end_constraint_type_ = DircolKinConstraintType::kPosOnly;
  force_cost_ = 1.0e-4;
}

void DircolOptions::setAllConstraintsRelative(bool relative) {
  for (int i=0; i < n_constraints_; i++) {
    is_constraints_relative_[i] = relative;
  }
}

void DircolOptions::setConstraintRelative(int index, bool relative) {
  is_constraints_relative_[index]  = relative;
}

void DircolOptions::setPhiValue(int index, double val) {
  DRAKE_DEMAND(is_constraints_relative_[index] == false);
  phi_vals_[index] = val;
}

void DircolOptions::setStartType(DircolKinConstraintType type) {
  start_constraint_type_ = type;
}

void DircolOptions::setEndType(DircolKinConstraintType type) {
  end_constraint_type_ = type;
}

void DircolOptions::setForceCost(double force_cost) {
  force_cost_ = force_cost;
}

int DircolOptions::getNumConstraints() {
  return n_constraints_;
}

bool DircolOptions::getSingleConstraintRelative(int index) {
  return is_constraints_relative_[index];
}

std::vector<bool> DircolOptions::getConstraintsRelative() {
  return is_constraints_relative_;
}

double DircolOptions::getSinglePhiValue(int index) {
  return phi_vals_[index];
}

drake::VectorX<double> DircolOptions::getPhiValues() {
  return phi_vals_;
}

DircolKinConstraintType DircolOptions::getStartType() {
  return start_constraint_type_;
}

DircolKinConstraintType DircolOptions::getEndType() {
  return end_constraint_type_;
}

double DircolOptions::getForceCost() {
  return force_cost_;
}

int DircolOptions::getNumRelative() {
  return static_cast<int>(std::count(is_constraints_relative_.begin(),
                                     is_constraints_relative_.end(), true));
}

}  // namespace trajectory_optimization
}  // namespace systems
}  // namespace dairlib

#pragma once

#include <vector>

#include "systems/trajectory_optimization/dircol/dircol_opt_constraints.h"

namespace dairlib {
namespace systems {
namespace trajectory_optimization {

class DircolOptions {
 private:
  int n_constraints_;
  std::vector<bool> is_constraints_relative_;
  drake::VectorX<double> phi_vals_;
  DircolKinConstraintType start_constraint_type_;
  DircolKinConstraintType end_constraint_type_;
  double force_cost_;

 public:
  explicit DircolOptions(int n_constraints);

  void setAllConstraintsRelative(bool relative);
  void setConstraintRelative(int index, bool relative);
  void setPhiValue(int index, double val);
  void setStartType(DircolKinConstraintType type);
  void setEndType(DircolKinConstraintType type);
  void setForceCost(double force_cost);

  int getNumConstraints();
  bool getSingleConstraintRelative(int index);
  std::vector<bool> getConstraintsRelative();
  double getSinglePhiValue(int index);
  drake::VectorX<double> getPhiValues();
  DircolKinConstraintType getStartType();
  DircolKinConstraintType getEndType();
  double getForceCost();
  int getNumRelative();
};

}  // namespace trajectory_optimization
}  // namespace systems
}  // namespace dairlib

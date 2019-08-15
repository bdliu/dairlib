#pragma once

#include <memory>

#include "drake/solvers/constraint.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "systems/trajectory_optimization/dircol/dircol_kinematic_data.h"

namespace dairlib {

template <typename T>
class DircolPositionData : public DircolKinematicData<T> {
 public:
  DircolPositionData(const drake::multibody::MultibodyPlant<T>& plant,
    const drake::multibody::Body<T>& body,
    Eigen::Vector3d pt, bool isXZ = false,
    Eigen::Vector2d ground_rp = Eigen::Vector2d::Zero(),
    bool isYZ = false);
  ~DircolPositionData();

  // The workhorse function, updates and caches everything needed by the
  // outside world
  void updateConstraint(const drake::systems::Context<T>& context);

  void addFixedNormalFrictionConstraints(Eigen::Vector3d normal, double mu);

 private:
    const drake::multibody::Body<T>& body_;
    Eigen::Vector3d pt_;
    bool isXZ_;
    Eigen::Matrix<double, 2, 3> TXZ_;
    Eigen::Matrix3d inv_rot_mat_ground_;
    Eigen::Matrix<double, 2, 3> TXZ_and_ground_incline_;
};
}  // namespace dairlib

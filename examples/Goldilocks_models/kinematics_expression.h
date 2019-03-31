#pragma once

#include <iostream>
#include <string>
#include <vector>
#include "math.h"
#include <Eigen/Dense>
#include "drake/math/autodiff_gradient.h"
#include "drake/common/eigen_types.h"

#include "multibody/multibody_utils.h"

#include "drake/common/drake_assert.h"

using Eigen::Matrix;
using Eigen::Dynamic;

using std::cout;
using std::endl;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;

using drake::MatrixX;
using drake::VectorX;
using drake::AutoDiffXd;
using drake::math::DiscardGradient;

using drake::multibody::MultibodyPlant;

namespace dairlib {
namespace goldilocks_models {

template <typename T>
class KinematicsExpression {
 public:
  explicit KinematicsExpression(int n_s, int n_feature);
  explicit KinematicsExpression(int n_s, int n_feature,
      const MultibodyPlant<T> * plant);
  KinematicsExpression() {}  // Default constructor

  int getDimFeature();

  template <typename U, typename V>
  VectorX<T> getExpression(const U & theta, const V & q) const;

  template <typename U>
  VectorX<U> getFeature(const VectorX<U> & q) const;

 private:
  const MultibodyPlant<T> * plant_;
  int n_feature_;
  int n_q_;
  int n_s_;
  const std::vector<std::string> leg_link_names_{
    "left_upper_leg_mass", "left_lower_leg_mass",
    "right_upper_leg_mass", "right_lower_leg_mass"};
  Vector3d mass_disp_;
  Vector3d foot_disp_;
};

}  // namespace goldilocks_models
}  // namespace dairlib

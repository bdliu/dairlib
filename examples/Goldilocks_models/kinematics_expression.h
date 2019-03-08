#include <iostream>
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
  explicit KinematicsExpression(int n_z, int n_feature);
  explicit KinematicsExpression(int n_z, int n_feature,
      const MultibodyPlant<T> * plant);
  KinematicsExpression() {}  // Default constructor

  int getDimFeature();

  template <typename U, typename V>
  VectorX<T> getExpression(const U & theta, const V & x) const;

  template <typename U>
  VectorX<U> getFeature(const VectorX<U> & x) const;

 private:
  const MultibodyPlant<T> * plant_;
  int n_feature_;
  int n_x_;
  int n_z_;
};

}  // namespace goldilocks_models
}  // namespace dairlib


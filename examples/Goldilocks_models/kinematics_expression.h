#include <iostream>
#include "math.h"
#include <Eigen/Dense>
#include "drake/math/autodiff_gradient.h"
#include "drake/common/eigen_types.h"

using Eigen::Matrix;
using Eigen::Dynamic;

using std::cout;
using std::endl;
using Eigen::VectorXd;

using drake::MatrixX;
using drake::VectorX;
using drake::AutoDiffXd;
using drake::math::DiscardGradient;


namespace dairlib {
namespace goldilocks_models {

class KinematicsExpression {
 public:
  explicit KinematicsExpression(int n_z, int n_feature);

  int getDimFeature();

  template <typename U, typename V>
  VectorX<AutoDiffXd> getExpression(U theta, V x);

  template <typename T>
  T getFeature(T x);

 private:
  int n_feature_;
  int n_x_;
  int n_z_;
};

}  // namespace goldilocks_models
}  // namespace dairlib


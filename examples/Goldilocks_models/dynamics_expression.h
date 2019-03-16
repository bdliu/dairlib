#include <iostream>
#include "math.h"
#include <Eigen/Dense>
#include "drake/math/autodiff_gradient.h"
#include "drake/common/eigen_types.h"

#include "drake/common/drake_assert.h"

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

class DynamicsExpression {
 public:
  explicit DynamicsExpression(int n_zDDot, int n_featureDot);
  DynamicsExpression() {}  // Default constructor

  int getDimFeature();

  template <typename U, typename V>
  V getExpression(const U & theta, const V & z) const;

  template <typename T>
  T getFeature(const T & z) const;

 private:
  int n_featureDot_;
  int n_x_;
  int n_zDDot_;
};

}  // namespace goldilocks_models
}  // namespace dairlib


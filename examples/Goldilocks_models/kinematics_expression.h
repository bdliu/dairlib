#include <iostream>
#include <Eigen/Dense>
#include "drake/math/autodiff_gradient.h"
#include "drake/common/eigen_types.h"

using Eigen::Matrix;
using Eigen::Dynamic;

using std::cout;
using std::endl;
using Eigen::VectorXd;

using drake::AutoDiffXd;
using drake::MatrixX;
using drake::VectorX;



namespace dairlib {
namespace goldilocks_models {

class KinematicsExpression {
 public:
  explicit KinematicsExpression(int n_z);

  int getDimFeature();

  template <typename T>
  T getExpression(T & theta, T & x);

  template <typename T>
  T getFeature(T & x);

 private:
  int n_feature;
  int n_x_;
  int n_z_;
};

}  // namespace goldilocks_models
}  // namespace dairlib


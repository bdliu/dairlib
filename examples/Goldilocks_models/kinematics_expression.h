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

  template <typename T>
  MatrixX<T> & getExpression(MatrixX<T> & theta,
                                        MatrixX<T> & x);

  template <typename T>
  T getFeature(T & x);



  // VectorXd getFeature(VectorXd & x);



  template <typename T>
  int getDimFeature(MatrixX<T>& x);

 private:
  int n_feature_;
  int n_x_;
  int n_z_;

};

}  // namespace goldilocks_models
}  // namespace dairlib


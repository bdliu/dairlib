#include <Eigen/Dense>


using Eigen::Matrix;
using Eigen::Dynamic;

namespace dairlib {
namespace goldilocks_models {

class KinematicsExpression {

 public:
  explicit KinematicsExpression(int n_z);

  template <typename T>
  Matrix<T, Dynamic, 1> & getExpression(Matrix<T, Dynamic, 1> & theta,
                                        Matrix<T, Dynamic, 1> & x);

  template <typename T>
  Matrix<T, Dynamic, 1> & getFeature(Matrix<T, Dynamic, 1> & x);

  template <typename T>
  int getDimFeature(Matrix<T, Dynamic, 1>& x);

 private:
  int n_feature_;
  int n_x_;
  int n_z_;

};

}  // namespace goldilocks_models
}  // namespace dairlib


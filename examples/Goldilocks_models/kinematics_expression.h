#include <Eigen/Dense>


using Eigen::Matrix;
using Eigen::Dynamic;

namespace dairlib {
namespace goldilocks_models {

template <typename T>
class KinematicsExpression {

 public:
  explicit KinematicsExpression(int n_z);

  Matrix<T, Dynamic, 1> & getExpression(Matrix<T, Dynamic, 1> & theta,
                                        Matrix<T, Dynamic, 1> & x);

  Matrix<T, Dynamic, 1> & getFeature(Matrix<T, Dynamic, 1> & x);

  int getDimFeature(Matrix<T, Dynamic, 1>& x);

 private:
  int n_feature_;
  int n_x_;
  int n_z_;

};

}  // namespace goldilocks_models
}  // namespace dairlib


#include <iostream>
#include <string>
#include "math.h"
#include <Eigen/Dense>

using std::cout;
using std::endl;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;


int main() {

  // Reference: https://eigen.tuxfamily.org/dox/group__LeastSquares.html
  MatrixXd A = MatrixXd::Random(3, 2);
  VectorXd b = VectorXd::Random(3);
  cout << "The solution using normal equations is:\n"
     << (A.transpose() * A).ldlt().solve(A.transpose() * b) << endl;


  MatrixXd B = MatrixXd::Random(3, 2);
  cout << "The solution using normal equations is:\n"
     << (A.transpose() * A).ldlt().solve(A.transpose() * B) << endl;

  return 0;
}

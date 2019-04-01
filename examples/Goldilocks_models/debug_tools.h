#include "drake/common/trajectories/piecewise_polynomial.h"
#include "systems/goldilocks_models/file_utils.h"

#include <Eigen/Dense>
#include <iostream>
#include <string>

using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Matrix3Xd;
using drake::trajectories::PiecewisePolynomial;
using std::vector;
using std::shared_ptr;
using std::cout;
using std::endl;
using std::string;
using std::to_string;

namespace dairlib {
namespace goldilocks_models  {

// Create cubic splines from s and sdot
PiecewisePolynomial<double> createCubicSplineGivenSAndSdot(vector<VectorXd> h_vec,
    vector<VectorXd> s_vec, vector<VectorXd> ds_vec);

// Store splines in csv file for plotting
// The first row is time, and the rest rows are s
void storeSplineOfS(vector<VectorXd> h_vec,
    PiecewisePolynomial<double> s_spline,
    string directory, string prefix);

// Check whether your cubic spline implemented in dynamics constriant is correct
void checkSplineOfS(vector<VectorXd> h_vec, vector<VectorXd> dds_vec,
    PiecewisePolynomial<double> s_spline);

}  // namespace goldilocks_models
} // dairlib


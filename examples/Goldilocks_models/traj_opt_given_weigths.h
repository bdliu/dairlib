#pragma once

#include <string>
#include <Eigen/Dense>
#include "systems/trajectory_optimization/hybrid_dircon.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

namespace dairlib {
namespace goldilocks_models  {

void trajOptGivenWeights(
    int n_z, int n_zDot, int n_featureZ, int n_featureZDot,
    VectorXd & thetaZ, VectorXd & thetaZDot,
    double stride_length, double duration, int max_iter,
    std::string directory, std::string init_file, std::string output_prefix,
    vector<VectorXd> & w_sol_vec,
    vector<MatrixXd> & A_vec, vector<MatrixXd> & H_vec,
    vector<VectorXd> & y_vec,
    vector<VectorXd> & lb_vec, vector<VectorXd> & ub_vec,
    vector<VectorXd> & b_vec,
    vector<MatrixXd> & B_vec,
    const double & Q_double, const double & R_double,
    double epsilon);

}  // namespace goldilocks_models
}  // namespace dairlib

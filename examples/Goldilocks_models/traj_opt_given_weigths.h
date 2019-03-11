#pragma once

#include <string>
#include <Eigen/Dense>
#include "systems/trajectory_optimization/hybrid_dircon.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace dairlib {
namespace goldilocks_models  {

void trajOptGivenWeights(
    int n_z, int n_zDot, int n_featureZ, int n_featureZDot,
    VectorXd & thetaZ, VectorXd & thetaZDot,
    double stride_length, double duration, int max_iter,
    std::string directory, std::string init_file, std::string output_prefix,
    VectorXd & w_sol,
    MatrixXd & A, MatrixXd & H,
    VectorXd & y, VectorXd & lb, VectorXd & ub, VectorXd & b,
    MatrixXd & B);

}  // namespace goldilocks_models
}  // namespace dairlib

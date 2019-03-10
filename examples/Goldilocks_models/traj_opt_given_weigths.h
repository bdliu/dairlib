#pragma once

#include <string>
#include <Eigen/Dense>
#include "systems/trajectory_optimization/hybrid_dircon.h"

namespace dairlib {
namespace goldilocks_models  {

void trajOptGivenWeights(
    int n_z, int n_zDot, int n_featureZ, int n_featureZDot,
    Eigen::VectorXd & thetaZ, Eigen::VectorXd & thetaZDot,
    double stride_length, double duration, int iter, std::string directory,
    std::string init_file, std::string output_prefix);

}  // namespace goldilocks_models
}  // namespace dairlib

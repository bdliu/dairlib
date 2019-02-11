#pragma once

#include <string>
#include "systems/trajectory_optimization/hybrid_dircon.h"

namespace dairlib {
namespace goldilocks_models  {

void trajOptGivenWeights(
    double stride_length, double duration, int iter, std::string directory,
    std::string init_file, std::string weights_file, std::string output_prefix);

}  // namespace goldilocks_models
}  // namespace dairlib

#include "examples/Goldilocks_models/traj_opt_given_weigths.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::string;

namespace dairlib {
namespace goldilocks_models {

void findGoldilocksModels() {

      double stride_length = 0.3;
      double duration = 1;
      int iter = 200;
      string directory = "";
      string init_file = "";
      string weights_file = "";
      string output_prefix = "";

      trajOptGivenWeights(stride_length, duration, iter, 
                         directory, init_file,  weights_file, output_prefix); 

      std::cout<<"end\n";
}
}  // namespace goldilocks_models
}  // namespace dairlib

int main() {

  dairlib::goldilocks_models::findGoldilocksModels();
}

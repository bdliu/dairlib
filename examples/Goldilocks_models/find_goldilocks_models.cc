#include "examples/Goldilocks_models/traj_opt_given_weigths.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::string;

namespace dairlib {
namespace goldilocks_models {

void findGoldilocksModels() {

      double stride_length = 0.3;
      double duration = .5;
      int iter = 500;
      string directory = "examples/Goldilocks_models/data/";
      string init_file = "";
      // string init_file = "z.csv";
      string weights_file = "";
      string output_prefix = "";




      trajOptGivenWeights(stride_length, duration, iter, 
                         directory, init_file,  weights_file, output_prefix); 





}
}  // namespace goldilocks_models
}  // namespace dairlib

int main() {

  dairlib::goldilocks_models::findGoldilocksModels();
}

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
  // string init_file = "w.csv";
  string output_prefix = "";



  // parameters
  int n_z = 2;
  int n_zDot = n_z; // Assume that are the same (no quaternion)
  int n_featureZ = 2; // This should match with the dimension of the feature,
                      // since we are hard coding it now. (same below)
  int n_featureZDot = 2;
  int n_thetaZ = n_z * n_featureZ;
  int n_thetaZDot = (n_zDot/2) * n_featureZDot;
      // Assuming position and velocity has the same dimension
      // for the reduced order model.

  VectorXd thetaZ(n_thetaZ);
  VectorXd thetaZDot(n_thetaZDot);
  thetaZ = VectorXd::Zero(n_thetaZ);
  thetaZ(0) = 1;
  thetaZ(3) = 1;
  thetaZDot = VectorXd::Zero(n_thetaZDot);


  // declare constarint class for z here

  // declare constarint class for zDot here


  // The following function should return the solution x
  trajOptGivenWeights(n_z, n_zDot, n_featureZ, n_featureZDot, thetaZ, thetaZDot,
                      stride_length, duration, iter,
                      directory, init_file, output_prefix);

  // Construct the outer loop optimization based on the solutino x




}
}  // namespace goldilocks_models
}  // namespace dairlib

int main() {
  dairlib::goldilocks_models::findGoldilocksModels();
}

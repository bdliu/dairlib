#include <gflags/gflags.h>
#include <Eigen/Dense>
#include <string>

#include "systems/goldilocks_models/file_utils.h"
#include "drake/common/drake_assert.h"

#include "examples/Goldilocks_models/dynamics_expression.h"

#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/multibody/parsing/parser.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::string;
using std::to_string;
using std::cin;
using std::cout;
using std::endl;

namespace dairlib {
namespace goldilocks_models {

DEFINE_int32(iter, 0, "Which iteration");
DEFINE_int32(n_s, 2, "dimension of the RoM");
DEFINE_int32(num_linear_term, 4, "# of linear terms");
DEFINE_int32(num_quadratic_term, 10, "# of quadratic terms");
DEFINE_double(scaling_factor, -1, "Scaling factor");

DEFINE_bool(is_active_model, true, "Model has input?");
DEFINE_int32(n_tau, 1, "dimension of the input of the RoM");
DEFINE_int32(num_traj_opt_knots, 20, "# of traj opt knot points");
DEFINE_int32(num_batch, 5, "total number of batch");

// Terminal command samples:
//  ./bazel-bin/examples/Goldilocks_models/scale_theta --iter=211 --num_linear_term=4 --num_quadratic_term=10 --scaling_factor=0.2 --n_s=2 --num_traj_opt_knots=20 --n_tau=1
//  ./bazel-bin/examples/Goldilocks_models/scale_theta --iter=212 --num_linear_term=4 --num_quadratic_term=10 --scaling_factor=0.2 --n_s=2 --is_active_model=false

// Assumptions:
// 1. The order of the features listed in the order of power #
//    e.g. constant, s (and/or ds), s^2 (and/or ds^2, s*ds), ...
// 2. The highest order is quadratic
// 3. The model is control-affine
// 4. The input are at the end of the decision variables. (You can improve this
//    by comparing the values in csv.)
int doMain(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  cout << "iter = " << FLAGS_iter << endl;
  cout << "n_s = " << FLAGS_n_s << endl;
  // cout << "num_linear_term = " << FLAGS_num_linear_term << endl;
  // cout << "num_quadratic_term = " << FLAGS_num_quadratic_term << endl;
  cout << "scaling_factor = " << FLAGS_scaling_factor << endl;

  if (FLAGS_is_active_model) {
    cout << "The model has input.\n";
    cout << "n_tau = " << FLAGS_n_tau << endl;
    cout << "num_traj_opt_knots = " << FLAGS_num_traj_opt_knots << endl;
    cout << "num_batch = " << FLAGS_num_batch << endl;
  } else {
    cout << "The model doesn't have input.\n";
  }

  cout << "Are the above numbers correct? (Y/N)\n";
  char answer[1];
  cin >> answer;
  if (!((answer[0] == 'Y') || (answer[0] == 'y'))) {
    cout << "Ending the program, since the numbers are incorrect.\n";
    return 0;
  } else {
    cout << "Updating the parameters.\n";
  }

  const string directory = "examples/Goldilocks_models/find_models/data/";
  string prefix = directory + to_string(FLAGS_iter);

  // Read in the parameters for setup
  VectorXd theta_s = readCSV(prefix + string("_theta_s.csv")).col(0);
  VectorXd theta_sDDot = readCSV(prefix + string("_theta_sDDot.csv")).col(0);

  // Create MBP for setup
  MultibodyPlant<double> plant;
  Parser parser(&plant);
  std::string full_name = FindResourceOrThrow(
                            "examples/Goldilocks_models/PlanarWalkerWithTorso.urdf");
  parser.AddModelFromFile(full_name);
  plant.AddForceElement<drake::multibody::UniformGravityFieldElement>(
    -9.81 * Eigen::Vector3d::UnitZ());
  plant.WeldFrames(
    plant.world_frame(), plant.GetFrameByName("base"),
    drake::math::RigidTransform<double>());
  plant.Finalize();

  // Setup
  KinematicsExpression<double> kin_expression(FLAGS_n_s, 0, &plant);
  DynamicsExpression dyn_expression(FLAGS_n_s, 0);
  VectorXd dummy_q = VectorXd::Zero(plant.num_positions());
  VectorXd dummy_s = VectorXd::Ones(FLAGS_n_s);
  int n_feature_s = kin_expression.getFeature(dummy_q).size();
  int n_feature_sDDot = dyn_expression.getFeature(dummy_s, dummy_s).size();
  DRAKE_DEMAND(theta_sDDot.rows() == FLAGS_n_s * n_feature_sDDot);

  // Create a list of idx corresponding to const, linear, quadratic terms...
  std::vector<int> const_term_list;
  std::vector<int> linear_term_list;
  std::vector<int> quadratic_term_list;
  double k_test_scale = 1.3;
  VectorXd original_feat = dyn_expression.getFeature(dummy_s, dummy_s).size();
  VectorXd scaled_feat = dyn_expression.getFeature(
                           k_test_scale * dummy_s, k_test_scale * dummy_s).size();
  for (int i = 0; i < n_feature_sDDot; i++) {
    if (scaled_feat(i) == original_feat(i)) {
      const_term_list.push_back(i);
    } else if (scaled_feat(i) == k_test_scale * original_feat(i)) {
      linear_term_list.push_back(i);
    } else if (scaled_feat(i) == pow(k_test_scale, 2) * original_feat(i)) {
      quadratic_term_list.push_back(i);
    } else {
      cout << "No matching scaling for index #" << i << endl;
      return 0;
    }
  }
  cout << "\nConstant terms: ";
  for (int i : const_term_list)
    cout << i << ", ";
  cout << "\nLinear terms: ";
  for (int i : linear_term_list)
    cout << i << ", ";
  cout << "\nQuadratic terms: ";
  for (int i : quadratic_term_list)
    cout << i << ", ";

  // Read in the parameters
  VectorXd theta_s = readCSV(prefix + string("_theta_s.csv")).col(0);
  VectorXd theta_sDDot = readCSV(prefix + string("_theta_sDDot.csv")).col(0);

  // iterate through each element of s
  for (int s_row = 0; s_row < n_s; s_row++) {
    // Get the scaling factor automatically unless specified by the user
    double scaling_factor;
    if (FLAGS_scaling_factor != -1) {
      scaling_factor = FLAGS_scaling_factor;
    } else {
      // Get scaling factor
      double theta_sum = 0;
      for (int j = 0; j < n_feature_s; j++) {
        theta_sum += theta_s(j + s_row * n_feature_s);
      }
      scaling_factor = 1 / theta_sum;
    }

    // Scaling of kinematics parameters
    theta_s.segment(s_row * n_feature_s, n_feature_s) *= scaling_factor;


    // TODO: below is not done yet
    // Scaling of dynamics parameters
    int idx = 0;
    // constant term
    for (int i = 0; i < FLAGS_n_s; i++)
      theta_sDDot.segment(idx + i * n_feature_sDDot, 1) *= scaling_factor;
    idx += 1;
    // linear terms
    for (int i = 0; i < FLAGS_n_s; i++)
      theta_sDDot.segment(idx + i * n_feature_sDDot, FLAGS_num_linear_term) *= 1;
    idx += FLAGS_num_linear_term;
    // quadratic terms
    for (int i = 0; i < FLAGS_n_s; i++)
      theta_sDDot.segment(idx + i * n_feature_sDDot, FLAGS_num_quadratic_term) /=
        scaling_factor;
    idx += FLAGS_num_quadratic_term;

    // Store (overwrite) the parameters
    writeCSV(prefix + string("_theta_s.csv"), theta_s);
    writeCSV(prefix + string("_theta_sDDot.csv"), theta_sDDot);

    cout << "Finshed updating the parameters.\n";

    // Input part
    if (FLAGS_is_active_model) {
      cout << "Updating the inputs.\n";

      for (int batch = 0; batch < FLAGS_num_batch; batch++) {
        prefix = directory + to_string(FLAGS_iter) +  "_" + to_string(batch);
        VectorXd w = readCSV(prefix + string("_w.csv")).col(0);

        w.tail(FLAGS_n_tau * FLAGS_num_traj_opt_knots) *= scaling_factor;

        writeCSV(prefix + string("_w.csv"), w);
      }
      cout << "Finished updating the inputs.\n";
    }



  }  // end for (each row of s)

  return 0;
}

}  // namespace goldilocks_models
}  // namespace dairlib

int main(int argc, char* argv[]) {
  dairlib::goldilocks_models::doMain(argc, argv);
  return 0;
}

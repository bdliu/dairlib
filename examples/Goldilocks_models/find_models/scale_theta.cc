#include <gflags/gflags.h>
#include <Eigen/Dense>
#include <string>

#include "systems/goldilocks_models/file_utils.h"
#include "drake/common/drake_assert.h"

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
DEFINE_double(scaling_factor, 1, "Scaling factor");

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
// 4. The input are at the end of the decision variables.
int doMain(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  cout << "iter = " << FLAGS_iter << endl;
  cout << "n_s = " << FLAGS_n_s << endl;
  cout << "num_linear_term = " << FLAGS_num_linear_term << endl;
  cout << "num_quadratic_term = " << FLAGS_num_quadratic_term << endl;
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

  int n_feature_sDDot = 1 + FLAGS_num_linear_term + FLAGS_num_quadratic_term;

  // Read in the parameters
  VectorXd theta_s = readCSV(prefix + string("_theta_s.csv")).col(0);
  VectorXd theta_sDDot = readCSV(prefix + string("_theta_sDDot.csv")).col(0);
  DRAKE_DEMAND(theta_sDDot.rows() == FLAGS_n_s * n_feature_sDDot);

  // Scaling of kinematics parameters
  theta_s = theta_s * FLAGS_scaling_factor;
  // Scaling of dynamics parameters
  int idx = 0;
  // constant term
  for (int i = 0; i < FLAGS_n_s; i++)
    theta_sDDot.segment(idx + i * n_feature_sDDot, 1) *= FLAGS_scaling_factor;
  idx += 1;
  // linear terms
  for (int i = 0; i < FLAGS_n_s; i++)
    theta_sDDot.segment(idx + i * n_feature_sDDot, FLAGS_num_linear_term) *= 1;
  idx += FLAGS_num_linear_term;
  // quadratic terms
  for (int i = 0; i < FLAGS_n_s; i++)
    theta_sDDot.segment(idx + i * n_feature_sDDot, FLAGS_num_quadratic_term) /=
      FLAGS_scaling_factor;
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

      w.tail(FLAGS_n_tau * FLAGS_num_traj_opt_knots) *= FLAGS_scaling_factor;

      writeCSV(prefix + string("_w.csv"), w);
    }
    cout << "Finished updating the inputs.\n";
  }

  return 0;
}

}  // namespace goldilocks_models
}  // namespace dairlib

int main(int argc, char* argv[]) {
  dairlib::goldilocks_models::doMain(argc, argv);
  return 0;
}

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
DEFINE_int32(num_linear_term, 0, "# of linear terms");
DEFINE_int32(num_quadratic_term, 0, "# of quadratic terms");
DEFINE_double(scaling_factor, 1, "Scaling factor");

// Assume the order of the features listed in the order of power #
// e.g. constant, s (and/or ds), s^2 (and/or ds^2, s*ds), ...

// Also assume that the highest order is quadratic
int doMain(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  cout << "iter = " << FLAGS_iter << endl;
  cout << "num_linear_term = " << FLAGS_num_linear_term << endl;
  cout << "num_quadratic_term = " << FLAGS_num_quadratic_term << endl;
  cout << "scaling_factor = " << FLAGS_scaling_factor << endl;

  cout << "Are the above numbers correct? (Y/N)\n";
  char answer[1];
  cin >> answer;
  if (!((answer[0] == 'Y') || (answer[0] == 'y'))) {
    cout << "Ending the program, since the numbers are incorrect.\n";
    return 0;
  } else {
    cout << "The numbers are correct. Updating the parameters.\n";
  }

  const string directory = "examples/Goldilocks_models/data/";
  const string prefix = directory + to_string(FLAGS_iter);

  // Read in the parameters
  VectorXd theta_s = readCSV(prefix + string("_theta_s.csv")).col(0);
  VectorXd theta_sDDot = readCSV(prefix + string("_theta_sDDot.csv")).col(0);
  DRAKE_DEMAND(theta_sDDot.rows() ==
               1 + FLAGS_num_linear_term + FLAGS_num_quadratic_term);

  // Scaling of kinematics parameters
  theta_s = theta_s * FLAGS_scaling_factor;
  // Scaling of dynamics parameters
  int idx = 0;
  // constant term
  theta_sDDot(idx) = theta_sDDot(idx) * FLAGS_scaling_factor;
  idx += 1;
  // linear terms
  for (int i = idx; i < idx + FLAGS_num_linear_term; i++)
    theta_sDDot(i) = theta_sDDot(i);
  idx += FLAGS_num_linear_term;
  // quadratic terms
  for (int i = idx; i < idx + FLAGS_num_quadratic_term; i++)
    theta_sDDot(i) = theta_sDDot(i) / FLAGS_scaling_factor;
  idx += FLAGS_num_quadratic_term;

  // Store (overwrite) the parameters
  writeCSV(prefix + string("_theta_s.csv"), theta_s);
  writeCSV(prefix + string("_theta_sDDot.csv"), theta_sDDot);

  cout << "Finshed updating the parameters.\n";
  return 0;
}

}  // namespace goldilocks_models
}  // namespace dairlib

int main(int argc, char* argv[]) {
  dairlib::goldilocks_models::doMain(argc, argv);
  return 0;
}

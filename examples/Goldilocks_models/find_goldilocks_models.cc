#include "examples/Goldilocks_models/traj_opt_given_weigths.h"
#include "systems/goldilocks_models/file_utils.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::string;
using std::cout;
using std::endl;

namespace dairlib {
namespace goldilocks_models {

void findGoldilocksModels() {
  double stride_length = 0.3;
  double duration = .5;
  int max_inner_iter = 500;
  int max_outer_iter = 20;
  string directory = "examples/Goldilocks_models/data/";
  // string init_file = "";
  string init_file = "w0.csv";
  string output_prefix = "";
  int n_batch = 1;

  // Parameters
  int n_z = 2;
  int n_zDot = n_z; // Assume that are the same (no quaternion)
  int n_featureZ = 1; // This should match with the dimension of the feature,
                      // since we are hard coding it now. (same below)
  int n_featureZDot = 1;
  int n_thetaZ = n_z * n_featureZ;
  int n_thetaZDot = (n_zDot / 2) * n_featureZDot;
      // Assuming position and velocity has the same dimension
      // for the reduced order model.

  // Initial guess of theta
  VectorXd thetaZ(n_thetaZ);
  VectorXd thetaZDot(n_thetaZDot);
  thetaZ = VectorXd::Zero(n_thetaZ);
  // thetaZ(0) = 1;
  // thetaZ(3) = 1;
  thetaZDot = VectorXd::Zero(n_thetaZDot);

  // Vectors/Matrices for the outer loop
  std::vector<MatrixXd> A_vec;
  std::vector<MatrixXd> B_vec;
  std::vector<MatrixXd> H_vec;
  std::vector<MatrixXd> A_active_vec;
  std::vector<MatrixXd> B_active_vec;
  std::vector<VectorXd> lb_vec;
  std::vector<VectorXd> ub_vec;
  std::vector<VectorXd> y_vec;
  std::vector<VectorXd> b_vec;
  std::vector<VectorXd> w_sol_vec;
  std::vector<VectorXd> theta_vec;

  // Trajectory optimization with fixed model paramters
  trajOptGivenWeights(n_z, n_zDot, n_featureZ, n_featureZDot, thetaZ, thetaZDot,
                      stride_length, duration, max_inner_iter,
                      directory, init_file, output_prefix,
                      w_sol_vec, A_vec, H_vec,
                      y_vec, lb_vec, ub_vec, b_vec, B_vec);
  VectorXd theta(n_thetaZ+n_thetaZDot);
  theta << thetaZ, thetaZDot;
  theta_vec.push_back(theta);

  // Construct the outer loop optimization based on the solution w
  for (int iter = 1; iter <= max_outer_iter; iter++) {
    int input_batch = iter == 1 ? 1 : n_batch;

    // Construct vector/matrices for the optmization
    std::vector<double> nw_vec;
    std::vector<double> nl_vec;

    int nw=0,nt=0,nl=0;

    for (int batch = 0; batch < input_batch; batch++) {

      DRAKE_ASSERT(b_vec[batch].cols() == 1);
      DRAKE_ASSERT(lb_vec[batch].cols() == 1);
      DRAKE_ASSERT(ub_vec[batch].cols() == 1);
      DRAKE_ASSERT(y_vec[batch].cols() == 1);
      DRAKE_ASSERT(b_vec[batch].cols() == 1);
      DRAKE_ASSERT(w_sol_vec[batch].cols() == 1);

      int n_active = 0;
      double tol = 1e-4;
      for (int i = 0; i < y_vec[batch].rows(); i++) {
        if (y_vec[batch](i) >= ub_vec[batch](i) - tol ||
            y_vec[batch](i) <= lb_vec[batch](i) + tol)
          n_active++;
      }

      int nw_i = A_vec[batch].cols();
      int nt_i = B_vec[batch].cols();

      MatrixXd A_active(n_active, nw_i);
      MatrixXd B_active(n_active, nt_i);
      MatrixXd AB_active(n_active, nw_i + nt_i);

      int nl_i = 0;
      for (int i = 0; i < y_vec[batch].rows(); i++) {
        if (y_vec[batch](i) >= ub_vec[batch](i) - tol ||
            y_vec[batch](i) <= lb_vec[batch](i) + tol) {
          A_active.row(nl_i) = A_vec[batch].row(i);
          B_active.row(nl_i) = B_vec[batch].row(i);
          AB_active.row(nl_i) << A_vec[batch].row(i), B_vec[batch].row(i);
          nl_i++;
        }
      }

      A_active_vec.push_back(A_active);
      B_active_vec.push_back(B_active);
      nl_vec.push_back(nl_i);
      nw_vec.push_back(nw_i);

      nl += nl_i;
      nw += nw_i;
      if (batch == 0) {
        nt = nt_i;
      } else {
        DRAKE_ASSERT(nt == nt_i);
        DRAKE_ASSERT((theta_vec[0] - theta_vec[batch]).norm() == 0);
      }
    }



    std::cout << "***********Next iteration*************" << std::endl;

    // Clear the matrices for outer loop
    A_vec.clear();
    B_vec.clear();
    H_vec.clear();
    A_active_vec.clear();
    B_active_vec.clear();
    lb_vec.clear();
    ub_vec.clear();
    y_vec.clear();
    b_vec.clear();
    w_sol_vec.clear();
    theta_vec.clear();

    for(int batch = 0; batch < n_batch; batch++) {

      // some setup for each batch

      trajOptGivenWeights(n_z, n_zDot, n_featureZ, n_featureZDot,
                          thetaZ, thetaZDot,
                          stride_length, duration, max_inner_iter,
                          directory, init_file, output_prefix,
                          w_sol_vec, A_vec, H_vec,
                          y_vec, lb_vec, ub_vec, b_vec, B_vec);

    }
  }





}
}  // namespace goldilocks_models
}  // namespace dairlib

int main() {
  dairlib::goldilocks_models::findGoldilocksModels();
}

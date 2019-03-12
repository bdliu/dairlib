#include "examples/Goldilocks_models/traj_opt_given_weigths.h"
#include "systems/goldilocks_models/file_utils.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXcd;
using std::string;
using std::vector;
using std::cout;
using std::endl;

namespace dairlib {
namespace goldilocks_models {

MatrixXd solveInvATimesB(const MatrixXd & A, const MatrixXd & B) {
  MatrixXd X = (A.transpose() * A).ldlt().solve(A.transpose() * B);
  MatrixXd abs_resid = (A*X-B).cwiseAbs();
  VectorXd left_one = VectorXd::Ones(abs_resid.rows());
  VectorXd right_one = VectorXd::Ones(abs_resid.cols());
  cout << "residual-sum: "<< left_one.transpose()*abs_resid*right_one << endl;
  return X;
}
// MatrixXd solveInvATimesB(const MatrixXd & A, const VectorXd & b) {
//   MatrixXd X = (A.transpose() * A).ldlt().solve(A.transpose() * b);
//   cout << "residual-norm: "<< (A*X-b).norm() << endl;
//   return X;
// }

void findGoldilocksModels() {
  string directory = "examples/Goldilocks_models/data/";
  // string init_file = "";
  string init_file = "w0.csv";
  string output_prefix = "";

  // Parametres for tasks
  int n_batch = 1;
  double stride_length = 0.3;
  double duration = 0.746; // Fix the duration now since we add cost ourselves

  // Paramters for the inner loop optimization
  int max_inner_iter = 500;
  double R = 10;  // Cost on input effort
  double Q_double = 10; // Cost on velocity

  // Paramters for the outer loop optimization
  int max_outer_iter = 1;
  double threshold = 1e-4;
  double h_step = 1e-2;
  double epsilon = 1e-4;

  // Reduced order model parameters
  int n_z = 2;
  int n_zDot = n_z; // Assume that are the same (no quaternion)
  int n_featureZ = 1;    // n_feature should match with the dim of the feature,
  int n_featureZDot = 1; // since we are hard coding it now. (same below)
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
  vector<MatrixXd> A_vec;
  vector<MatrixXd> B_vec;
  vector<MatrixXd> H_vec;
  vector<MatrixXd> A_active_vec;
  vector<MatrixXd> B_active_vec;
  vector<VectorXd> lb_vec;
  vector<VectorXd> ub_vec;
  vector<VectorXd> y_vec;
  vector<VectorXd> y_active_vec;
  vector<VectorXd> b_vec;
  vector<VectorXd> w_sol_vec;
  vector<VectorXd> theta_vec;

  // Start the gradient descent
  VectorXd theta(n_thetaZ + n_thetaZDot);
  theta << thetaZ, thetaZDot;
  for (int iter = 1; iter <= max_outer_iter; iter++)  {
    cout << "*********** Iteration " << iter << " *************" << endl;
    int current_batch = iter == 1 ? 1 : n_batch;

    // Clear the vectors/matrices before trajectory optimization
    A_vec.clear();
    B_vec.clear();
    H_vec.clear();
    A_active_vec.clear();
    B_active_vec.clear();
    lb_vec.clear();
    ub_vec.clear();
    y_vec.clear();
    y_active_vec.clear();
    b_vec.clear();
    w_sol_vec.clear();
    theta_vec.clear();

    // Run trajectory optimization for different tasks first
    for (int batch = 0; batch < current_batch; batch++) {

      /// some setup for each batch


      // Trajectory optimization with fixed model paramters
      trajOptGivenWeights(n_z, n_zDot, n_featureZ, n_featureZDot,
                          thetaZ, thetaZDot,
                          stride_length, duration, max_inner_iter,
                          directory, init_file, output_prefix,
                          w_sol_vec, A_vec, H_vec,
                          y_vec, lb_vec, ub_vec, b_vec, B_vec,
                          Q_double, R,
                          epsilon);
      theta_vec.push_back(theta);

    }

    // Then do outer loop optimization given the solution w

    // Construct vectors/matrices for the problem (get the active constraints)
    vector<double> nw_vec;  // size of decision var of traj opt for all tasks
    vector<double> nl_vec;  // # of rows of active constraints for all tasks
    int nw = 0;  // sum of size of decision variables for all task
    int nt = 0;  // size of parameters theta
    int nl = 0;  // sum of # of rows of active constraints for all tasks
    for (int batch = 0; batch < current_batch; batch++) {
      DRAKE_ASSERT(b_vec[batch].cols() == 1);
      DRAKE_ASSERT(lb_vec[batch].cols() == 1);
      DRAKE_ASSERT(ub_vec[batch].cols() == 1);
      DRAKE_ASSERT(y_vec[batch].cols() == 1);
      DRAKE_ASSERT(b_vec[batch].cols() == 1);
      DRAKE_ASSERT(w_sol_vec[batch].cols() == 1);

      int nw_i = A_vec[batch].cols();
      int nt_i = B_vec[batch].cols();

      int nl_i = 0;
      double tol = 1e-4;
      for (int i = 0; i < y_vec[batch].rows(); i++) {
        if (y_vec[batch](i) >= ub_vec[batch](i) - tol ||
            y_vec[batch](i) <= lb_vec[batch](i) + tol)
          nl_i++;
      }

      nw_vec.push_back(nw_i);
      nl_vec.push_back(nl_i);

      nw += nw_i;
      nl += nl_i;

      MatrixXd A_active(nl_i, nw_i);
      MatrixXd B_active(nl_i, nt_i);
      VectorXd y_active(nl_i);

      nl_i = 0;
      for (int i = 0; i < y_vec[batch].rows(); i++) {
        if (y_vec[batch](i) >= ub_vec[batch](i) - tol ||
            y_vec[batch](i) <= lb_vec[batch](i) + tol) {
          A_active.row(nl_i) = A_vec[batch].row(i);
          B_active.row(nl_i) = B_vec[batch].row(i);
          y_active(nl_i) = y_vec[batch](i);
          nl_i++;
        }
      }

      A_active_vec.push_back(A_active);
      B_active_vec.push_back(B_active);
      y_active_vec.push_back(y_active);

      if (batch == 0) {
        nt = nt_i;
      } else {
        DRAKE_ASSERT(nt == nt_i);
        DRAKE_ASSERT((theta_vec[0] - theta_vec[batch]).norm() == 0);
      }
    }

    cout << "nw = " << nw << endl;
    cout << "nt = " << nt << endl;
    cout << "nl = " << nl << endl;

    // Reference for solving a sparse linear system
    // https://eigen.tuxfamily.org/dox/group__TopicSparseSystems.html
    // https://eigen.tuxfamily.org/dox/group__LeastSquares.html

    // Our calculation below is based on the fact that the H matrices are pd and
    // symmetric, so we check them here.
    // However, H turned out not to be psd, since we have timestep h as decision
    // variable. (It came from running cost. ~h*u'*R*u, etc)
    // Fixed it by adding running cost by hand (but the timestep is fixed now).
    cout << "Checking if H is pd and symmetric\n";
    for (int batch = 0; batch < current_batch; batch++) {
      // Check if H is symmetric
      VectorXd One_w = VectorXd::Ones(nw_vec[batch]);
      double sum =
        One_w.transpose() * (H_vec[batch] - H_vec[batch].transpose()) * One_w;
      if (sum != 0) cout << "H is not symmetric\n";

      // Check if H is pd
      VectorXd eivals_real = H_vec[batch].eigenvalues().real();
      for (int i = 0; i < eivals_real.size(); i++) {
        if (eivals_real(i) <= 0)
          cout << "H is not positive definite (with e-value = "
               << eivals_real(i) << ")\n";
      }
    }
    cout << "Finished checking\n";


    // Testing
    // Eigen::BDCSVD<MatrixXd> svd(H_vec[0]);
    // int n_sv = svd.singularValues().size();
    // cout << "smallest singular value is " << svd.singularValues()(n_sv-1) << endl;


    // Get P_i and q_i
    // Method 1: use optimization program to solve it???
    // Method 2: use schur complement (see notes)
    // This one requires the Hessian H to be pd.
    // (w = P_i * theta + q_i)
    vector<MatrixXd> P_vec;
    vector<VectorXd> q_vec;
    for (int batch = 0; batch < current_batch; batch++) {
      MatrixXd AinvQA = A_active_vec[batch] * solveInvATimesB(
                          H_vec[batch], A_active_vec[batch].transpose());
      VectorXd invQc = solveInvATimesB(H_vec[batch], b_vec[batch]);
      MatrixXd E = solveInvATimesB(AinvQA, B_active_vec[batch]);
      VectorXd F = -solveInvATimesB(AinvQA, y_active_vec[batch]+A_active_vec[batch]*invQc);


      // Testing
      // Eigen::BDCSVD<MatrixXd> svd(H_vec[0]);
      // int n_sv = svd.singularValues().size();
      // cout << "smallest singular value is " << svd.singularValues()(n_sv-1) << endl;



      MatrixXd Pi = -solveInvATimesB(H_vec[batch],A_active_vec[batch].transpose()*E);
      VectorXd qi = -solveInvATimesB(H_vec[batch],b_vec[batch]+A_active_vec[batch].transpose()*F);

      cout << "qi norm = " << qi.norm() << endl;
      P_vec.push_back(Pi);
      q_vec.push_back(qi);
    }

    // Get gradient of the cost wrt theta (assume H_vec[batch] symmetric)
    VectorXd costGradient = VectorXd::Zero(theta_vec[0].size());
    for (int batch = 0; batch < current_batch; batch++) {
      costGradient +=
        P_vec[batch].transpose()*(b_vec[batch]+H_vec[batch]*q_vec[batch]);
    }

    // Gradient descent and assign thetaZ and thetaZDot
    theta -= h_step * costGradient;
    thetaZ = theta.head(n_thetaZ);
    thetaZDot = theta.tail(n_thetaZDot);

    // Check optimality
    cout << "costGradient norm: " << costGradient.norm() << endl;
    if(costGradient.norm() < threshold){
      cout << "Found optimal theta.\n";
      break;
    }
  }





}
}  // namespace goldilocks_models
}  // namespace dairlib

int main() {
  dairlib::goldilocks_models::findGoldilocksModels();
}

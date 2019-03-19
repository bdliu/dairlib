#include "examples/Goldilocks_models/traj_opt_given_weigths.h"
#include "systems/goldilocks_models/file_utils.h"

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/solve.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::to_string;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXcd;
using drake::solvers::MathematicalProgram;

namespace dairlib {
namespace goldilocks_models {

MatrixXd solveInvATimesB(const MatrixXd & A, const MatrixXd & B) {
  MatrixXd X = (A.transpose() * A).ldlt().solve(A.transpose() * B);
  MatrixXd abs_resid = (A * X - B).cwiseAbs();
  VectorXd left_one = VectorXd::Ones(abs_resid.rows());
  VectorXd right_one = VectorXd::Ones(abs_resid.cols());
  cout << "sum-abs-residual: " << left_one.transpose()*abs_resid*right_one <<
       endl;
  return X;
}
// MatrixXd solveInvATimesB(const MatrixXd & A, const VectorXd & b) {
//   MatrixXd X = (A.transpose() * A).ldlt().solve(A.transpose() * b);
//   cout << "residual-norm: "<< (A*X-b).norm() << endl;
//   return X;
// }

void findGoldilocksModels() {
  const string directory = "examples/Goldilocks_models/data/";
  string init_file;
  // init_file = "";
  init_file = "w0.csv";
  // init_file = "w0_with_z.csv";
  string prefix = "";

  // Parametres for tasks
  int n_batch = 5;//1;
  double delta_stride_length = 0.03;
  double stride_length_0 = 0.3;
  double duration = 0.746; // Fix the duration now since we add cost ourselves
  vector<double> delta_stride_length_vec;
  for (int i = 0 - n_batch / 2; i < n_batch - n_batch / 2; i++)
    delta_stride_length_vec.push_back(i * delta_stride_length);

  // Paramters for the outer loop optimization
  int iter_start = 265;
  int max_outer_iter = 10000;
  double threshold = 1e-4;
  double h_step = 1e-2;  // 1e-1 caused divergence when close to optimal sol
  double eps_regularization = 1e-4;
  bool is_newton = true;

  // Paramters for the inner loop optimization
  int max_inner_iter = 500;
  double R = 10;  // Cost on input effort
  double Q_double = 10; // Cost on velocity

  // Reduced order model parameters
  int n_s = 1; //2
  int n_sDDot = n_s; // Assume that are the same (no quaternion)
  int n_feature_s =
    113;//1;//113    // n_feature should match with the dim of the feature,
  int n_feature_sDDot =
    7;//1;//21 // since we are hard coding it now. (same below)
  int n_theta_s = n_s * n_feature_s;
  int n_theta_sDDot = n_sDDot * n_feature_sDDot;
  // Assuming position and velocity has the same dimension
  // for the reduced order model.

  // Initial guess of theta
  VectorXd theta_s(n_theta_s);
  VectorXd theta_sDDot(n_theta_sDDot);
  // theta_s = VectorXd::Zero(n_theta_s);
  // theta_s(0) = 1;
  // theta_sDDot = VectorXd::Zero(n_theta_sDDot);
  // theta_sDDot(0) = 1;
  // // Testing intial theta
  // theta_s = VectorXd::Random(n_theta_s);
  // theta_sDDot = VectorXd::Random(n_theta_sDDot);
  MatrixXd theta_s_mat =
    readCSV(directory + to_string(iter_start) + string("_theta_s.csv"));
  MatrixXd theta_sDDot_mat =
    readCSV(directory + to_string(iter_start) + string("_theta_sDDot.csv"));
  theta_s = theta_s_mat.col(0);
  theta_sDDot = theta_sDDot_mat.col(0);

  // Vectors/Matrices for the outer loop
  vector<VectorXd> w_sol_vec;
  vector<MatrixXd> H_vec;
  vector<VectorXd> b_vec;
  vector<VectorXd> c_vec;
  vector<MatrixXd> A_vec;
  vector<MatrixXd> A_active_vec;
  vector<VectorXd> lb_vec;
  vector<VectorXd> ub_vec;
  vector<VectorXd> y_vec;
  vector<VectorXd> y_active_vec;
  vector<MatrixXd> B_vec;
  vector<MatrixXd> B_active_vec;

  //
  double min_so_far = 10000000;

  // Start the gradient descent
  int n_theta = n_theta_s + n_theta_sDDot;
  VectorXd theta(n_theta);
  theta << theta_s, theta_sDDot;
  VectorXd prev_theta = theta;
  VectorXd step_direction;
  for (int iter = iter_start; iter <= max_outer_iter; iter++)  {
    cout << "*********** Iteration " << iter << " *************" << endl;
    if (iter != 0) cout << "theta_sDDot = " << theta_sDDot.transpose() << endl;

    // setup for each iteration
    bool is_get_nominal = iter == 0 ? true : false;
    int current_batch = is_get_nominal ? 1 : n_batch;
    int max_inner_iter_pass_in = is_get_nominal ? 1000 : max_inner_iter;

    // store initial parameter values
    if (is_get_nominal) {
      prefix = to_string(iter) +  "_";
      writeCSV(directory + prefix + string("theta_s.csv"), theta_s);
      writeCSV(directory + prefix + string("theta_sDDot.csv"), theta_sDDot);
    }

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
    c_vec.clear();
    w_sol_vec.clear();

    // Run trajectory optimization for different tasks first
    for (int batch = 0; batch < current_batch; batch++) {
      /// setup for each batch
      double stride_length = is_get_nominal ? stride_length_0 :
                             stride_length_0 + delta_stride_length_vec[batch];
      cout << "stride_length = " << stride_length << endl;
      prefix = to_string(iter) +  "_" + to_string(batch) + "_";
      string init_file_pass_in;
      if (is_get_nominal)
        init_file_pass_in = init_file;
      else if (iter == 1)
        init_file_pass_in = string("0_0_w.csv");
      else
        init_file_pass_in = to_string(iter - 1) +  "_" +
                            to_string(batch) + string("_w.csv");

      // Trajectory optimization with fixed model paramters
      trajOptGivenWeights(n_s, n_sDDot, n_feature_s, n_feature_sDDot,
                          theta_s, theta_sDDot,
                          stride_length, duration, max_inner_iter_pass_in,
                          directory, init_file_pass_in, prefix,
                          w_sol_vec, A_vec, H_vec,
                          y_vec, lb_vec, ub_vec, b_vec, c_vec, B_vec,
                          Q_double, R,
                          eps_regularization,
                          is_get_nominal);
    }
    double total_cost = 0;
    for (int batch = 0; batch < current_batch; batch++)
      total_cost += c_vec[batch](0) / current_batch;
    if (total_cost <= min_so_far) {
      min_so_far = total_cost;
    }
    cout << "total_cost = " << total_cost << " (min so far: " << min_so_far <<
         ")\n\n";

    // If the cost goes up, shrink the size and redo the traj opt.
    // Otherwise, do outer loop optimization given the solution w.
    if (total_cost > min_so_far) {
      h_step = h_step / 2;
      cout << "Step size shrinks to " << h_step << ". Redo this iteration.\n\n";
      iter -= 1;

      // testing
      cout << "previous theta_sDDot = " << prev_theta.tail(n_theta_sDDot).transpose() << endl;

      // Descent
      theta = prev_theta + h_step * step_direction;

      // Assign theta_s and theta_sDDot
      theta_s = theta.head(n_theta_s);
      theta_sDDot = theta.tail(n_theta_sDDot);
      // store parameter values
      prefix = to_string(iter + 1) +  "_";
      writeCSV(directory + prefix + string("theta_s.csv"), theta_s);
      writeCSV(directory + prefix + string("theta_sDDot.csv"), theta_sDDot);
    }
    else if (!is_get_nominal) {
      // Extract active and independent constraints
      cout << "Extracting active and independent rows of A\n";
      vector<double> nw_vec;  // size of decision var of traj opt for all tasks
      vector<double> nl_vec;  // # of rows of active constraints for all tasks
      int nw = 0;  // sum of size of decision variables for all task
      int nl = 0;  // sum of # of rows of active constraints for all tasks
      for (int batch = 0; batch < current_batch; batch++) {
        DRAKE_ASSERT(b_vec[batch].cols() == 1);
        DRAKE_ASSERT(lb_vec[batch].cols() == 1);
        DRAKE_ASSERT(ub_vec[batch].cols() == 1);
        DRAKE_ASSERT(y_vec[batch].cols() == 1);
        DRAKE_ASSERT(w_sol_vec[batch].cols() == 1);

        int nt_i = B_vec[batch].cols();
        int nw_i = A_vec[batch].cols();
        nw_vec.push_back(nw_i);
        nw += nw_i;

        int nl_i = 0;
        double tol = 1e-4;
        for (int i = 0; i < y_vec[batch].rows(); i++) {
          if (y_vec[batch](i) >= ub_vec[batch](i) - tol ||
              y_vec[batch](i) <= lb_vec[batch](i) + tol)
            nl_i++;
        }

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




        /*cout << "\n (After extracting active constraints) Run traj opt to check if your quadratic approximation is correct\n";
        nl_i = A_active.rows();
        nw_i = A_active.cols();
        MathematicalProgram quadprog;
        auto w2 = quadprog.NewContinuousVariables(nw_i, "w2");
        quadprog.AddLinearConstraint( A_active,
                                  VectorXd::Zero(nl_i),
                                  VectorXd::Zero(nl_i),
                                  w2);
        quadprog.AddQuadraticCost(H_vec[0],b_vec[0],w2);
        const auto result2 = Solve(quadprog);
        auto solution_result2 = result2.get_solution_result();
        cout << solution_result2 << endl;
        cout << "Cost:" << result2.get_optimal_cost() << endl;
        VectorXd w_sol_check = result2.GetSolution(quadprog.decision_variables());
        cout << "w_sol norm:" << w_sol_check.norm() << endl;
        cout << "Finished traj opt\n\n";

        cout << "This should be zero\n" << VectorXd::Ones(nl_i).transpose()*A_active*w_sol_check << endl;
        cout << "if this is not zero, then w=0 is not optimal: " << w_sol_check.transpose()*b_vec[0] << endl;*/






        /*// Find redundant rows
        cout << "Find redundant rows of constraints\n";
        vector<int> non_redundant_row_idx;
        non_redundant_row_idx.push_back(0);
        VectorXd rowi(nw_i);
        VectorXd rowj(nw_i);
        VectorXd normalized_rowi(nw_i);
        VectorXd normalized_rowj(nw_i);
        unsigned int count = 0; // see if it goes through all element of vector
        for (int i = 1; i < nl_i; i++) {
          count = 0;
          for (int j : non_redundant_row_idx) {
            rowi = A_active.row(i).transpose();
            rowj = A_active.row(j).transpose();
            normalized_rowi = rowi / rowi.norm();
            normalized_rowj = rowj / rowj.norm();
            if ((normalized_rowi - normalized_rowj).norm() < 1e-6) {
              cout << "There are redundant rows (" << j << "," << i << ")\n";
              // We don't need to check the b in Ax=b, because we know there are
              // feasible solutions
              // But we still check it just in case.
              if (y_active(i) / rowi.norm() - y_active(j) / rowj.norm() > 1e-6)
                cout << "There are over-constraining rows!!!!\n";
              // Checking if B is involved
              for(int k = 0; k<n_theta_s + n_theta_sDDot; k++){
                if(B_active(i,k) != 0) cout << "B is in redundant rows of constraints\n";
              }
              break;
            }
            count++;
          }
          if (count == non_redundant_row_idx.size())
            non_redundant_row_idx.push_back(i);
        }
        cout << "Finished finding redundant rows of constraints\n";

        nl_i = non_redundant_row_idx.size();

        // Get rid of redundant rows
        MatrixXd A_active_nonredundant(nl_i, nw_i);
        MatrixXd B_active_nonredundant(nl_i, nt_i);
        VectorXd y_active_nonredundant(nl_i);
        for (int i = 0; i < nl_i; i++) {
          A_active_nonredundant.row(i) = A_active.row(non_redundant_row_idx[i]);
          B_active_nonredundant.row(i) = B_active.row(non_redundant_row_idx[i]);
          y_active_nonredundant(i) = y_active(non_redundant_row_idx[i]);
        }*/

        // Only add the rows that are linearly independent
        // cout << "Start extracting independent rows of A\n";
        std::vector<int> full_row_rank_idx;
        full_row_rank_idx.push_back(0);
        for (int i = 1; i < nl_i; i++) {
          // Construct test matrix
          int n_current_rows = full_row_rank_idx.size();
          MatrixXd A_test(n_current_rows + 1, nw_i);
          for (unsigned int j = 0 ; j < full_row_rank_idx.size(); j++) {
            A_test.block(j, 0, 1, nw_i) =
              A_active.row(full_row_rank_idx[j]);
          }
          A_test.block(n_current_rows, 0, 1, nw_i) = A_active.row(i);

          // Perform svd to check rank
          Eigen::BDCSVD<MatrixXd> svd(A_test);
          // double sigular_value = svd.singularValues()(n_current_rows);
          if (svd.singularValues()(n_current_rows) > 1e-6) {
            full_row_rank_idx.push_back(i);
          }
        }
        // cout << "Finished extracting independent rows of A\n\n";

        nl_i = full_row_rank_idx.size();
        nl_vec.push_back(nl_i);
        nl += nl_i;

        // Assign the rows
        MatrixXd A_full_row_rank(nl_i, nw_i);
        MatrixXd B_full_row_rank(nl_i, nt_i);
        VectorXd y_full_row_rank(nl_i);
        for (int i = 0; i < nl_i; i++) {
          A_full_row_rank.row(i) = A_active.row(full_row_rank_idx[i]);
          B_full_row_rank.row(i) = B_active.row(full_row_rank_idx[i]);
          y_full_row_rank(i) = y_active(full_row_rank_idx[i]);
        }

        A_active_vec.push_back(A_full_row_rank);
        B_active_vec.push_back(B_full_row_rank);
        y_active_vec.push_back(y_full_row_rank);
      }  // end extracting active and independent constraints
      cout << endl;

      // cout << "nw = " << nw << endl;
      // cout << "nl = " << nl << endl;

      // Reference for solving a sparse linear system
      // https://eigen.tuxfamily.org/dox/group__TopicSparseSystems.html
      // https://eigen.tuxfamily.org/dox/group__LeastSquares.html

      // Our calculation below is based on the fact that the H matrices are pd and
      // symmetric, so we check them here.
      // However, H turned out not to be psd, since we have timestep h as decision
      // variable. (It came from running cost. ~h*u'*R*u, etc)
      // Fixed it by adding running cost by hand (but the timestep is fixed now).
      // Now H is always pd because we also added a regularization term.
      /*cout << "Checking if H is pd and symmetric\n";
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
      cout << "Finished checking\n\n";*/


      // Get w in terms of theta (Get P_i and q_i where w = P_i * theta + q_i)
      vector<MatrixXd> P_vec;
      vector<VectorXd> q_vec;
      P_vec.clear();
      q_vec.clear();
      // cout << "Getting P matrix and q vecotr\n";
      // Method 1: use optimization program to solve it???

      // Method 2: use schur complement (see notes)
      /*
      // This one requires the Hessian H to be pd.
      // Also, although this method might be more computationally efficient, the
      // accuracy is not as high as when we use inverse() directly. The reason is
      // that the condition number of A and invH is high, so AinvHA' makes it very
      // ill-conditioned.
      for (int batch = 0; batch < current_batch; batch++) {
        MatrixXd AinvHA = A_active_vec[batch] * solveInvATimesB(
                            H_vec[batch], A_active_vec[batch].transpose());
        VectorXd invQc = solveInvATimesB(H_vec[batch], b_vec[batch]);
        MatrixXd E = solveInvATimesB(AinvHA, B_active_vec[batch]);
        VectorXd F = -solveInvATimesB(AinvHA, A_active_vec[batch] * invQc);
        // Testing
        Eigen::BDCSVD<MatrixXd> svd(AinvHA);
        cout << "AinvHA':\n";
        cout << "  biggest singular value is " << svd.singularValues()(0) << endl;
        cout << "  smallest singular value is "
                << svd.singularValues().tail(1) << endl;
        cout << "The condition number of A and invH are large. That's why AinvHA'"
                "is ill-conditioned.\n";
        // cout << "singular values are \n" << svd.singularValues() << endl;

        MatrixXd Pi = -solveInvATimesB(H_vec[batch],
                                       A_active_vec[batch].transpose() * E);
        VectorXd qi = -solveInvATimesB(H_vec[batch],
                              b_vec[batch] + A_active_vec[batch].transpose() * F);
        cout << "qi norm (this number should be close to 0) = "
             << qi.norm() << endl;
        P_vec.push_back(Pi);
        q_vec.push_back(qi);
      }*/

      // Method 3: use inverse() directly
      // H_ext = [H A'; A 0]
      for (int batch = 0; batch < current_batch; batch++) {
        int nl_i = nl_vec[batch];
        int nw_i = nw_vec[batch];
        MatrixXd H_ext(nw_i + nl_i, nw_i + nl_i);
        H_ext.block(0, 0, nw_i, nw_i) = H_vec[batch];
        H_ext.block(0, nw_i, nw_i, nl_i) = A_active_vec[batch].transpose();
        H_ext.block(nw_i, 0, nl_i, nw_i) = A_active_vec[batch];
        H_ext.block(nw_i, nw_i, nl_i, nl_i) = MatrixXd::Zero(nl_i, nl_i);

        // cout << "\nStart inverting the matrix.\n";
        MatrixXd inv_H_ext = H_ext.inverse();
        // cout << "Finsihed inverting the matrix.\n";

        MatrixXd inv_H_ext11 = inv_H_ext.block(0, 0, nw_i, nw_i);
        MatrixXd inv_H_ext12 = inv_H_ext.block(0, nw_i, nw_i, nl_i);

        MatrixXd Pi = -inv_H_ext12 * B_active_vec[batch];
        VectorXd qi = -inv_H_ext11 * b_vec[batch];

        /*MatrixXd abs_Pi = Pi.cwiseAbs();
        VectorXd left_one = VectorXd::Ones(abs_Pi.rows());
        VectorXd right_one = VectorXd::Ones(abs_Pi.cols());
        cout << "sum-abs-Pi: " <<
             left_one.transpose()*abs_Pi*right_one << endl;
        cout << "sum-abs-Pi divide by m*n: " <<
             left_one.transpose()*abs_Pi*right_one / (abs_Pi.rows()*abs_Pi.cols())
             << endl;
        double max_Pi_element = abs_Pi(0, 0);
        for (int i = 0; i < abs_Pi.rows(); i++)
          for (int j = 0; j < abs_Pi.cols(); j++) {
            if (abs_Pi(i, j) > max_Pi_element) max_Pi_element = abs_Pi(i, j);
          }
        cout << "max element of abs-Pi = " << max_Pi_element << endl;
        cout << "qi norm (this number should be close to 0) = "
             << qi.norm() << endl;*/

        P_vec.push_back(Pi);
        q_vec.push_back(qi);
      }





      /*// Testing
      Eigen::BDCSVD<MatrixXd> svd(H_vec[0]);
      cout << "H:\n";
      cout << "  biggest singular value is " << svd.singularValues()(0) << endl;
      cout << "  smallest singular value is "
              << svd.singularValues().tail(1) << endl;
      // cout << "singular values are \n" << svd.singularValues() << endl;
      // Testing
      MatrixXd invH = H_vec[0].inverse();
      Eigen::BDCSVD<MatrixXd> svd_4(invH);
      cout << "invH:\n";
      cout << "  biggest singular value is " << svd_4.singularValues()(0) << endl;
      cout << "  smallest singular value is "
              << svd_4.singularValues().tail(1) << endl;
      // Testing
      Eigen::BDCSVD<MatrixXd> svd_2(A_active_vec[0]);
      cout << "A:\n";
      cout << "  biggest singular value is " << svd_2.singularValues()(0) << endl;
      cout << "  smallest singular value is "
              << svd_2.singularValues().tail(1) << endl;
      // cout << "singular values are \n" << svd_2.singularValues() << endl;

      // Testing
      Eigen::BDCSVD<MatrixXd> svd_3(H_ext);
      cout << "H_ext:\n";
      cout << "  biggest singular value is " << svd_3.singularValues()(0) << endl;
      cout << "  smallest singular value is "
              << svd_3.singularValues().tail(1) << endl;

      Eigen::BDCSVD<MatrixXd> svd_5(inv_H_ext);
      cout << "inv_H_ext:\n";
      cout << "  biggest singular value is " << svd_5.singularValues()(0) << endl;
      cout << "  smallest singular value is "
              << svd_5.singularValues().tail(1) << endl;*/




      // Run a quadprog to check if the solution to the following problem is 0
      // Theoratically, it should be 0. Otherwise, something is wrong
      // min 0.5*w^T Q w + c^T w
      // st  A w = 0
      /*cout << "Run traj opt to check if your quadratic approximation is correct\n";
      int nl_i = nl_vec[0];
      int nw_i = nw_vec[0];
      MathematicalProgram quadprog;
      auto w = quadprog.NewContinuousVariables(nw_i, "w");
      quadprog.AddLinearConstraint( A_active_vec[0],
                                VectorXd::Zero(nl_i),
                                VectorXd::Zero(nl_i),
                                w);
      quadprog.AddQuadraticCost(H_vec[0],b_vec[0],w);
      const auto result = Solve(quadprog);
      auto solution_result = result.get_solution_result();
      cout << solution_result << endl;
      cout << "Cost:" << result.get_optimal_cost() << endl;
      VectorXd w_sol = result.GetSolution(quadprog.decision_variables());
      cout << "w_sol norm:" << w_sol.norm() << endl;
      cout << "Finished traj opt\n\n";*/







      /*// Check if Q_theta is pd
      cout << "Checking if Q_theta is psd...\n";
      MatrixXd Q_theta = MatrixXd::Zero(n_theta, n_theta);
      for (int batch = 0; batch < current_batch; batch++)
        Q_theta += P_vec[batch].transpose()*H_vec[batch]*P_vec[batch];
      VectorXd eivals_real = Q_theta.eigenvalues().real();
      for (int i = 0; i < eivals_real.size(); i++) {
        if (eivals_real(i) <= 0)
          cout << "Q_theta is not positive definite (with e-value = "
               << eivals_real(i) << ")\n";
      }
      cout << endl;*/







      // Get gradient of the cost wrt theta (assume H_vec[batch] symmetric)
      // cout << "Calculating gradient\n";
      VectorXd costGradient = VectorXd::Zero(theta.size());
      for (int batch = 0; batch < current_batch; batch++) {
        costGradient += P_vec[batch].transpose() * b_vec[batch] / current_batch;
        // costGradient +=
        // P_vec[batch].transpose() * (b_vec[batch] + H_vec[batch] * q_vec[batch]);
      }
      // cout << "costGradient = \n" << costGradient;

      // Newton's method (not exactly the same, cause Q_theta is not pd but psd)
      // See your IOE611 lecture notes on page 7-17 to page 7-20
      // cout << "Getting Newton step\n";
      MatrixXd Q_theta = MatrixXd::Zero(n_theta, n_theta);
      for (int batch = 0; batch < current_batch; batch++) {
        Q_theta +=
          P_vec[batch].transpose() * H_vec[batch] * P_vec[batch] / current_batch;
      }
      double mu = 1e-4; // 1e-6 caused unstable and might diverge
      MatrixXd inv_Q_theta = (Q_theta + mu * MatrixXd::Identity(n_theta,
                              n_theta)).inverse();
      VectorXd newton_step = -inv_Q_theta * costGradient;
      // Testing
      /*Eigen::BDCSVD<MatrixXd> svd(inv_Q_theta);
      cout << "inv_Q_theta's smallest and biggest singular value " <<
           svd.singularValues().tail(1) << ", " <<
           svd.singularValues()(0) << endl;*/
      // Newton decrement (can be a criterion to terminate your newton steps)
      double lambda_square = -costGradient.transpose() * newton_step;

      // step_direction
      step_direction = is_newton ? newton_step : -costGradient;



      // TODO
      // If the cost goes up, or the problem is infeasible, just decrease the step size!



      // TODO(yminchen): only add the cost that has a optimal traj opt solution
      // Terminate the optimization if more than have samples don't have sol.






      // Gradient descent
      prev_theta = theta;
      cout << "theta_sDDot = " << theta.tail(n_theta_sDDot).transpose() << endl;
      if(is_newton)
        theta = theta + h_step * step_direction;
      else
        theta = theta + h_step * step_direction;

      // Assign theta_s and theta_sDDot
      theta_s = theta.head(n_theta_s);
      theta_sDDot = theta.tail(n_theta_sDDot);
      // store parameter values
      prefix = to_string(iter + 1) +  "_";
      writeCSV(directory + prefix + string("theta_s.csv"), theta_s);
      writeCSV(directory + prefix + string("theta_sDDot.csv"), theta_sDDot);

      // Check optimality
      cout << "lambda_square = " << lambda_square << endl;
      cout << "costGradient norm: " << costGradient.norm() << endl << endl;
      if(is_newton){
        if (lambda_square < threshold) {
          cout << "Found optimal theta.\n\n";
          break;
        }
      }
      else{
        if (costGradient.norm() < threshold) {
          cout << "Found optimal theta.\n\n";
          break;
        }
      }
    } // if(!is_get_nominal)
  } // end for



}
}  // namespace goldilocks_models
}  // namespace dairlib

int main() {
  dairlib::goldilocks_models::findGoldilocksModels();
}

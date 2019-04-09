#include <iostream>
#include <string>
#include "math.h"
#include <Eigen/Dense>

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/solve.h"
using drake::solvers::MathematicalProgram;
#include "solvers/optimization_utils.h"

using std::cout;
using std::endl;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::VectorXcd;

int main() {

  /*  // Example 1
    MatrixXd H_o(1,1);
    H_o << 1;
    VectorXd b_o(1);
    b_o << 0;
    MatrixXd A_o = 0.5*MatrixXd::Ones(1,1);
    VectorXd lb_o = 0.5*VectorXd::Ones(1);
    VectorXd ub_o = VectorXd::Ones(1);
    cout << "In the original problem: \n";
    cout << "H_o = " << H_o << endl;
    cout << "b_o = " << b_o << endl;
    cout << "A_o = " << A_o << endl;
    cout << "lb_o = " << lb_o << endl;
    cout << "ub_o = " << ub_o << endl;

    cout << "\nSolving optimization problem...\n";
    MathematicalProgram quadprog;
    auto w = quadprog.NewContinuousVariables(1, "w");
    quadprog.AddLinearConstraint(A_o,
                              lb_o,
                              ub_o,
                              w);
    // Adds a cost term of the form 0.5*x'*H_o*x + b_o'x + c
    quadprog.AddQuadraticCost(H_o,b_o,w);
    // quadprog.AddQuadraticCost(0.5*w(0)*H_o(0,0)*w(0) + w(0)*b_o(0));
    const auto result = Solve(quadprog);
    auto solution_result = result.get_solution_result();
    cout << solution_result << endl;
    cout << "Cost:" << result.get_optimal_cost() << endl;
    VectorXd w_sol = result.GetSolution(quadprog.decision_variables());
    cout << "w_sol:" << w_sol << endl;
    // cout << "w_sol norm:" << w_sol.norm() << endl;
    cout << "\n";


    // Approximation
    cout << "Approximated problem: \n";
    MatrixXd A, H;
    VectorXd y, lb, ub, b;
    dairlib::systems::trajectory_optimization::linearizeConstraints(
      &quadprog, w_sol, y, A, lb, ub);
    double c = dairlib::systems::trajectory_optimization::secondOrderCost(
      &quadprog, w_sol, H, b);

    cout << "H = " << H << endl;
    cout << "b = " << b << endl;
    cout << "A = " << A << endl;
    cout << "lb = " << lb << endl;
    cout << "ub = " << ub << endl;
    cout << "Optimal cost (c) = " << c << endl;
    cout << "y (constraint value) = " << y << endl;*/

  //////////////////////////////////////////////////////////////////////////////
  /*// Example 2
  MatrixXd rand_mat = MatrixXd::Random(2,2);
  MatrixXd H_o = rand_mat.transpose() + rand_mat;
  VectorXd b_o = VectorXd::Ones(2);
  MatrixXd A_o = 0.5*MatrixXd::Random(2,2);
  VectorXd lb_o = 0.5*VectorXd::Ones(2);
  VectorXd ub_o = VectorXd::Ones(2);
  cout << "In the original problem: \n";
  cout << "H_o = \n" << H_o << endl;
  cout << "b_o = " << b_o.transpose() << endl;
  cout << "c_o = 0\n";
  cout << "A_o = \n" << A_o << endl;
  cout << "lb_o = " << lb_o.transpose() << endl;
  cout << "ub_o = " << ub_o.transpose() << endl;

  cout << "\nSolving optimization problem...\n";
  MathematicalProgram quadprog;
  auto w = quadprog.NewContinuousVariables(2, "w");
  quadprog.AddLinearConstraint(A_o,
                            lb_o,
                            ub_o,
                            w);
  // Adds a cost term of the form 0.5*x'*H_o*x + b_o'x + c
  quadprog.AddQuadraticCost(H_o,b_o,w);
  // quadprog.AddQuadraticCost(0.5*w(0)*H_o(0,0)*w(0) + w(0)*b_o(0));
  const auto result = Solve(quadprog);
  auto solution_result = result.get_solution_result();
  cout << solution_result << endl;
  cout << "Cost:" << result.get_optimal_cost() << endl;
  VectorXd w_sol = result.GetSolution(quadprog.decision_variables());
  cout << "w_sol:" << w_sol.transpose() << endl;
  // cout << "w_sol norm:" << w_sol.norm() << endl;
  cout << "\n";

  // Approximation
  cout << "Approximated problem: \n";
  MatrixXd A, H;
  VectorXd y, lb, ub, b;
  dairlib::systems::trajectory_optimization::linearizeConstraints(
    &quadprog, w_sol, y, A, lb, ub);
  double c = dairlib::systems::trajectory_optimization::secondOrderCost(
    &quadprog, w_sol, H, b);

  cout << "H = \n" << H << endl;
  cout << "b = " << b.transpose() << endl;
  cout << "c (Optimal cost) = " << c << endl;
  cout << "A = \n" << A << endl;
  cout << "lb = " << lb.transpose() << endl;
  cout << "ub = " << ub.transpose() << endl;
  cout << "y (constraint value) = " << y.transpose() << endl;*/


  //////////////////////////////////////////////////////////////////////////////
  // Example 3
  MatrixXd H_o(2, 2);
  H_o <<  1.36075, 0.354964,
      0.354964,  1.19376;
  VectorXd b_o = VectorXd::Ones(2);
  MatrixXd A_o(2, 2);
  A_o <<  0.411647, -0.164777,
      -0.302449, 0.26823;
  VectorXd lb_o = 0.5 * VectorXd::Ones(2);
  VectorXd ub_o = VectorXd::Ones(2);
  cout << "In the original problem: \n";
  cout << "H_o = \n" << H_o << endl;
  cout << "b_o = " << b_o.transpose() << endl;
  cout << "c_o = 0\n";
  cout << "A_o = \n" << A_o << endl;
  cout << "lb_o = " << lb_o.transpose() << endl;
  cout << "ub_o = " << ub_o.transpose() << endl;

  cout << "\nSolving optimization problem...\n";
  MathematicalProgram quadprog;
  auto w = quadprog.NewContinuousVariables(2, "w");
  quadprog.AddLinearConstraint(A_o.row(0),
                               lb_o(0),
                               ub_o(0),
                               w);
  quadprog.AddLinearConstraint(A_o.row(1),
                               lb_o(1),
                               ub_o(1),
                               w);
  // Adds a cost term of the form 0.5*x'*H_o*x + b_o'x + c
  quadprog.AddQuadraticCost(H_o * 1 / 3, b_o * 1 / 3, w);
  quadprog.AddQuadraticCost(H_o * 2 / 3, b_o * 2 / 3, w);
  // quadprog.AddQuadraticCost(0.5*w(0)*H_o(0,0)*w(0) + w(0)*b_o(0));
  const auto result = Solve(quadprog);
  auto solution_result = result.get_solution_result();
  cout << solution_result << endl;
  cout << "Cost:" << result.get_optimal_cost() << endl;
  VectorXd w_sol = result.GetSolution(quadprog.decision_variables());
  cout << "w_sol:" << w_sol.transpose() << endl;
  // cout << "w_sol norm:" << w_sol.norm() << endl;
  cout << "\n";

  // Approximation
  cout << "Approximated problem: \n";
  MatrixXd A, H;
  VectorXd y, lb, ub, b;
  dairlib::systems::trajectory_optimization::linearizeConstraints(
    &quadprog, w_sol, y, A, lb, ub);
  double c = dairlib::systems::trajectory_optimization::secondOrderCost(
               &quadprog, w_sol, H, b);
  VectorXd c_vec(1); c_vec << c;

  cout << "H = \n" << H << endl;
  cout << "b = " << b.transpose() << endl;
  cout << "c (Optimal cost) = " << c << endl;
  cout << "A = \n" << A << endl;
  cout << "lb = " << lb.transpose() << endl;
  cout << "ub = " << ub.transpose() << endl;
  cout << "y (constraint value) = " << y.transpose() << endl;

  // Test the cost function
  std::vector<VectorXd> w_sample;
  w_sample.push_back(VectorXd::Random(2));
  w_sample.push_back(VectorXd::Random(2));
  w_sample.push_back(VectorXd::Random(2));
  cout << w_sample[0] << endl;
  cout << w_sample[1] << endl;
  cout << w_sample[2] << endl;

  VectorXd original_cost_value;
  VectorXd approx_cost_value;
  for(int i = 0; i<3; i++){
    original_cost_value = 0.5 * w_sample[i].transpose()*H_o*w_sample[i] + b_o.transpose()*w_sample[i];
    approx_cost_value = 0.5 * (w_sample[i]-w_sol).transpose()*H*(w_sample[i]-w_sol) + b.transpose()*(w_sample[i]-w_sol) + c_vec;
    cout << "original_cost_value-approx_cost_value = " << original_cost_value-approx_cost_value << endl;
  }


  return 0;
}

#pragma once

#include <string>
#include <Eigen/Dense>
#include "systems/trajectory_optimization/hybrid_dircon.h"

#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/constraint.h"
#include "drake/solvers/solve.h"

#include "multibody/multibody_utils.h"

using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
using drake::solvers::SolutionResult;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

using drake::multibody::MultibodyPlant;
using drake::AutoDiffXd;

namespace dairlib {
namespace goldilocks_models  {

void trajOptGivenWeights(
    const MultibodyPlant<double> & plant,
    const MultibodyPlant<AutoDiffXd> & plant_autoDiff,
    int n_s, int n_sDDot, int n_tau, int n_feature_s, int n_feature_sDDot,
    MatrixXd B_tau,
    const VectorXd & theta_s, const VectorXd & theta_sDDot,
    double stride_length, double ground_incline, double duration, int max_iter,
    std::string directory, std::string init_file, std::string prefix,
    /*vector<VectorXd> * w_sol_vec,
    vector<MatrixXd> * A_vec, vector<MatrixXd> * H_vec,
    vector<VectorXd> * y_vec,
    vector<VectorXd> * lb_vec, vector<VectorXd> * ub_vec,
    vector<VectorXd> * b_vec,
    vector<VectorXd> * c_vec,
    vector<MatrixXd> * B_vec,*/
    double Q_double, double R_double,
    double eps_reg,
    bool is_get_nominal,
    bool is_zero_touchdown_impact,
    bool extend_model,
    bool is_add_tau_in_cost,
    int sample);

}  // namespace goldilocks_models
}  // namespace dairlib

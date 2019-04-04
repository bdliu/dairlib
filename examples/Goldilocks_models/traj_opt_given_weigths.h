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

MathematicalProgramResult trajOptGivenWeights(
    MultibodyPlant<double> & plant,
    MultibodyPlant<AutoDiffXd> & plant_autoDiff,
    int n_s, int n_sDDot, int n_feature_s, int n_feature_sDDot,
    VectorXd & theta_s, VectorXd & theta_sDDot,
    double stride_length, double duration, int max_iter,
    std::string directory, std::string init_file, std::string prefix,
    vector<VectorXd> & w_sol_vec,
    vector<MatrixXd> & A_vec, vector<MatrixXd> & H_vec,
    vector<VectorXd> & y_vec,
    vector<VectorXd> & lb_vec, vector<VectorXd> & ub_vec,
    vector<VectorXd> & b_vec,
    vector<VectorXd> & c_vec,
    vector<MatrixXd> & B_vec,
    const double & Q_double, const double & R_double,
    double eps_reg,
    bool is_get_nominal);

}  // namespace goldilocks_models
}  // namespace dairlib

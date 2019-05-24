#include <gflags/gflags.h>
#include <chrono>
#include <ctime>

#include "systems/goldilocks_models/file_utils.h"

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/solve.h"

#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/multibody/parsing/parser.h"

#include "common/find_resource.h"

#include "examples/Goldilocks_models/kinematics_expression.h"
#include "examples/Goldilocks_models/dynamics_expression.h"

#include "examples/Goldilocks_models/planning/RoM_planning_traj_opt.h"

using std::cin;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::to_string;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::VectorXcd;
using Eigen::MatrixXd;
using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
using drake::solvers::SolutionResult;

using drake::geometry::SceneGraph;
using drake::multibody::MultibodyPlant;
using drake::multibody::Body;
using drake::multibody::Parser;
using drake::AutoDiffXd;
using dairlib::FindResourceOrThrow;

namespace dairlib {
namespace goldilocks_models {

DEFINE_int32(iter, 0, "The iteration # of the theta that you use");
DEFINE_string(init_file, "z0.csv", "Initial Guess for Planning Optimization");

// Planning with optimal reduced order model and full order model
// (discrete map is from full order model)
int planningWithRomAndFom(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Create MBP
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

  // Create autoDiff version of the plant
  MultibodyPlant<AutoDiffXd> plant_autoDiff(plant);

  // Files parameters
  const string dir_model = "examples/Goldilocks_models/planning/models/";
  const string dir_data = "examples/Goldilocks_models/planning/data/";
  string init_file = FLAGS_init_file;

  // Reduced order model parameters
  cout << "Warning: Need to make sure that the implementation in "
       "DynamicsExpression agrees with n_s and n_tau.\n";
  int n_s = 4;
  int n_sDDot = n_s;
  int n_tau = 2;
  cout << "n_s = " << n_s << ", n_tau = " << n_tau << endl;
  MatrixXd B_tau = MatrixXd::Zero(n_sDDot, n_tau);
  B_tau(2, 0) = 1;
  B_tau(3, 1) = 1;
  cout << "B_tau = \n" << B_tau << endl;

  // Reduced order model setup
  KinematicsExpression<double> kin_expression(n_s, 0, &plant);
  DynamicsExpression dyn_expression(n_sDDot, 0);
  VectorXd dummy_q = VectorXd::Zero(plant.num_positions());
  VectorXd dummy_s = VectorXd::Zero(n_s);
  int n_feature_s = kin_expression.getFeature(dummy_q).size();
  int n_feature_sDDot =
    dyn_expression.getFeature(dummy_s, dummy_s).size();
  int n_theta_s = n_s * n_feature_s;
  int n_theta_sDDot = n_sDDot * n_feature_sDDot;
  // cout << "n_theta_s = " << n_theta_s << endl;
  // cout << "n_theta_sDDot = " << n_theta_sDDot << endl;

  // Read in theta
  string dir_and_pf = dir_model + to_string(FLAGS_iter) + string("_");
  // cout << "dir_and_pf = " << dir_and_pf << endl;
  VectorXd theta_s = readCSV(dir_and_pf + string("theta_s.csv")).col(0);
  VectorXd theta_sDDot = readCSV(dir_and_pf + string("theta_sDDot.csv")).col(0);
  DRAKE_DEMAND(theta_s.size() == n_theta_s);
  DRAKE_DEMAND(theta_sDDot.size() == n_theta_sDDot);
  // cout << "theta_s = " << theta_s.transpose() << endl;
  // cout << "theta_sDDot = " << theta_sDDot.transpose() << endl;

  // Optimization parameters
  MatrixXd Q = MatrixXd::Identity(n_s, n_s);
  MatrixXd R = MatrixXd::Identity(n_tau, n_tau);

  // Prespecify the time steps
  std::vector<int> num_time_samples;
  num_time_samples.push_back(20);
  num_time_samples.push_back(20);
  std::vector<double> min_dt;
  min_dt.push_back(.01);
  min_dt.push_back(.01);
  std::vector<double> max_dt;
  max_dt.push_back(.3);
  max_dt.push_back(.3);
  int N = 0;
  for (uint i = 0; i < num_time_samples.size(); i++)
    N += num_time_samples[i];
  N -= num_time_samples.size() - 1;

  // Construct
  auto trajopt = std::make_unique<RomPlanningTrajOptWithFomImpactMap>(
                   num_time_samples, min_dt, max_dt,
                   n_s, n_tau, B_tau,
                   n_feature_s, n_feature_sDDot, theta_s, theta_sDDot, plant);

  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Print file", "snopt.out");
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Major iterations limit", 1000);
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Verify level", 0);

  cout << "Adding cost...\n";
  auto x = trajopt->state();
  auto u = trajopt->input();
  trajopt->AddRunningCost(u.transpose()*R * u);
  trajopt->AddRunningCost(x.tail(n_s).transpose()*Q * x.tail(n_s));

  // Initial guess
  if (!init_file.empty()) {
    VectorXd z0 = readCSV(dir_data + init_file).col(0);
    int n_dec = trajopt->decision_variables().size();
    if (n_dec > z0.rows()) {
      cout << "dim(initial guess) < dim(decision var). "
           "Fill the rest with zero's.\n";
      VectorXd old_z0 = z0;
      z0.resize(n_dec);
      z0 = VectorXd::Zero(n_dec);
      z0.head(old_z0.rows()) = old_z0;
    }
    trajopt->SetInitialGuessForAllVariables(z0);
  }

  // Solve
  cout << "Solving DIRCON (based on MultipleShooting)\n";
  auto start = std::chrono::high_resolution_clock::now();
  const MathematicalProgramResult result = Solve(
        *trajopt, trajopt->initial_guess());
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  cout << "    Solve time:" << elapsed.count() << " | ";
  SolutionResult solution_result = result.get_solution_result();
  cout << solution_result <<  " | ";
  cout << "Cost:" << result.get_optimal_cost() << ")\n";

  // Extract solution
  VectorXd z_sol = result.GetSolution(trajopt->decision_variables());
  writeCSV(dir_data + string("z.csv"), z_sol);

  // Store the time, state, and input at knot points
  VectorXd time_at_knots = trajopt->GetSampleTimes(result);
  MatrixXd state_at_knots = trajopt->GetStateSamples(result);
  MatrixXd input_at_knots = trajopt->GetInputSamples(result);
  writeCSV(dir_data + string("time_at_knots.csv"), time_at_knots);
  writeCSV(dir_data + string("state_at_knots.csv"), state_at_knots);
  writeCSV(dir_data + string("input_at_knots.csv"), input_at_knots);



  return 0;
}  // int planningWithRomAndFom
}  // namespace goldilocks_models
}  // namespace dairlib

int main(int argc, char* argv[]) {
  return dairlib::goldilocks_models::planningWithRomAndFom(argc, argv);
}

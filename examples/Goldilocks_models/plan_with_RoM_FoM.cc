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
DEFINE_string(init_file, "w0.csv", "Initial Guess for Planning Optimization");

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
  const string dir = "examples/Goldilocks_models/find_models/data/";
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
  cout << "n_theta_s = " << n_theta_s << endl;
  cout << "n_theta_sDDot = " << n_theta_sDDot << endl;

  // Read in theta
  string dir_prefix = dir + to_string(FLAGS_iter) + string("_");
  cout << "dir_prefix = " << dir_prefix << endl;
  VectorXd theta_s = readCSV(dir_prefix + string("theta_s.csv")).col(0);
  VectorXd theta_sDDot = readCSV(dir_prefix + string("theta_sDDot.csv")).col(0);
  DRAKE_DEMAND(theta_s.size() == n_theta_s);
  DRAKE_DEMAND(theta_sDDot.size() == n_theta_sDDot);
  cout << "theta_s = " << theta_s.transpose() << endl;
  cout << "theta_sDDot = " << theta_sDDot.transpose() << endl;

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

  return 0;
}  // int planningWithRomAndFom
}  // namespace goldilocks_models
}  // namespace dairlib

int main(int argc, char* argv[]) {
  return dairlib::goldilocks_models::planningWithRomAndFom(argc, argv);
}

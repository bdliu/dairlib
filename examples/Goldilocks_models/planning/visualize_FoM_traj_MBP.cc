#include <gflags/gflags.h>

#include <memory>
#include <chrono>

#include <string>

#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"

#include "drake/lcm/drake_lcm.h"

#include "drake/multibody/parsing/parser.h"
#include "drake/systems/rendering/multibody_position_to_geometry_pose.h"
#include "drake/geometry/geometry_visualization.h"

#include "common/find_resource.h"
#include "systems/primitives/subvector_pass_through.h"

#include "multibody/multibody_utils.h"
#include "multibody/visualization_utils.h"

#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/lcm/drake_lcm.h"

#include "drake/common/trajectories/piecewise_polynomial.h"

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/solve.h"

#include "systems/goldilocks_models/file_utils.h"

#include "examples/Goldilocks_models/planning/kinematics_constraint_given_r.h"
#include "examples/Goldilocks_models/planning/kinematics_constraint_only_pos.h"
#include "examples/Goldilocks_models/planning/FoM_stance_foot_constraint_given_pos.h"

using drake::multibody::MultibodyPlant;
using drake::geometry::SceneGraph;
using drake::multibody::Body;
using drake::multibody::Parser;
using drake::systems::rendering::MultibodyPositionToGeometryPose;

using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Matrix3Xd;
using drake::trajectories::PiecewisePolynomial;
using drake::MatrixX;
using std::vector;
using std::shared_ptr;
using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::to_string;

using drake::math::DiscardGradient;
using drake::math::autoDiffToValueMatrix;
using drake::math::autoDiffToGradientMatrix;
using drake::math::initializeAutoDiff;

namespace dairlib {
namespace goldilocks_models {
namespace planning {

DEFINE_int32(iter, 131, "Which iteration");
DEFINE_int32(n_mode, 5, "# of modes");
DEFINE_int32(n_nodes, 20, "# of nodes per mode");
DEFINE_int32(n_feature_kin, 70, "# of kinematics features");
DEFINE_double(step_time, 1, "Duration per step * 2");
DEFINE_bool(is_do_inverse_kin, false, "Do inverse kinematics for presentation");

map<string, int> doMakeNameToPositionsMap() {
  // Create a plant
  drake::systems::DiagramBuilder<double> builder;
  MultibodyPlant<double> plant;
  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  Parser parser(&plant, &scene_graph);
  std::string full_name = FindResourceOrThrow(
                            "examples/Goldilocks_models/PlanarWalkerWithTorso.urdf");
  parser.AddModelFromFile(full_name);
  plant.AddForceElement<drake::multibody::UniformGravityFieldElement>(
    -9.81 * Eigen::Vector3d::UnitZ());
  plant.WeldFrames(
    plant.world_frame(), plant.GetFrameByName("base"),
    drake::math::RigidTransform<double>());
  plant.Finalize();

  return multibody::makeNameToPositionsMap(plant);
}

// This program visualizes the full order model by doing inverse kinematics.
// This program is used with the planning which contains FoM at pre/post impact.
// Therefore, this program is just for presentation usage. (so that easy to
// explain to other what you are doing)
void visualizeFullOrderModelTraj(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // parameters
  const string dir_data = "examples/Goldilocks_models/planning/data/";

  bool start_with_left_stance = true;
  if (start_with_left_stance)
    cout << "This program assumes the robot start with left stance. ";
  else
    cout << "This program assumes the robot start with right stance. ";
  cout << "Is this correct? (Y/N)\n";
  char answer[1];
  cin >> answer;
  if (!((answer[0] == 'Y') || (answer[0] == 'y'))) {
    cout << "Ending the program.\n";
    return;
  } else {
    cout << "Continue the program...\n";
  }
  bool left_stance = start_with_left_stance;

  cout << "Total # of modes = " << FLAGS_n_mode << endl;
  cout << "# of nodes per mode = " << FLAGS_n_nodes << endl;
  cout << "n_feature_kin = " << FLAGS_n_feature_kin << endl;
  cout << "iter (used for theta) = " << FLAGS_iter << endl;

  cout << "Are the above numbers correct? (Y/N)\n";
  char answer2[1];
  cin >> answer2;
  if (!((answer2[0] == 'Y') || (answer2[0] == 'y'))) {
    cout << "Ending the program, since the numbers are incorrect.\n";
    return;
  } else {
    cout << "Continue the program...\n";
  }

  // Read in the parameters
  const string dir_model = "examples/Goldilocks_models/planning/models/";
  string prefix = dir_model + to_string(FLAGS_iter);
  VectorXd theta_kin = readCSV(prefix + string("_theta_s.csv")).col(0);

  // Read in pose
  MatrixXd x0_each_mode =
    goldilocks_models::readCSV(dir_data + string("x0_each_mode.csv"));
  MatrixXd xf_each_mode =
    goldilocks_models::readCSV(dir_data + string("xf_each_mode.csv"));
  DRAKE_DEMAND(x0_each_mode.cols() == FLAGS_n_mode);

  // Read in states
  MatrixXd states =
    goldilocks_models::readCSV(dir_data + string("state_at_knots.csv"));
  DRAKE_DEMAND(states.cols() == FLAGS_n_mode * FLAGS_n_nodes -
               (FLAGS_n_mode - 1));
  int n_r = states.rows() / 2;

  // Get the stance foot positions
  MatrixXd stance_foot_pos_each_mode(2, x0_each_mode.cols());
  for (int i = 0; i < FLAGS_n_mode; i++) {
    if (start_with_left_stance)
      left_stance = (i % 2) ? false : true;
    else
      left_stance = (i % 2) ? true : false;

    VectorX<double> q0 = x0_each_mode.col(i).head(7);
    if (left_stance) {
      VectorX<double> left_foot_pos_xz_0(2);
      left_foot_pos_xz_0 <<
                         q0(0) - 0.5 * sin(q0(2) + q0(3)) - 0.5 * sin(q0(2) + q0(3) + q0(5)),
                         q0(1) - 0.5 * cos(q0(2) + q0(3)) - 0.5 * cos(q0(2) + q0(3) + q0(5));
      stance_foot_pos_each_mode.col(i) = left_foot_pos_xz_0;
    } else {
      VectorX<double> right_foot_pos_xz_0(2);
      right_foot_pos_xz_0 <<
                          q0(0) - 0.5 * sin(q0(2) + q0(4)) - 0.5 * sin(q0(2) + q0(4) + q0(6)),
                          q0(1) - 0.5 * cos(q0(2) + q0(4)) - 0.5 * cos(q0(2) + q0(4) + q0(6));
      stance_foot_pos_each_mode.col(i) = right_foot_pos_xz_0;
    }
  }
  cout << "stance_foot_pos_each_mode = \n" << stance_foot_pos_each_mode << endl;

  // Some setup
  int n_q = 7;
  map<string, int> positions_map = doMakeNameToPositionsMap();
  MatrixXd q_at_all_knots(7, states.cols());
  for (int i = 0; i < FLAGS_n_mode; i++) {
    q_at_all_knots.col((FLAGS_n_nodes - 1) * i) = x0_each_mode.col(i);
    q_at_all_knots.col((FLAGS_n_nodes - 1) * (i + 1)) = xf_each_mode.col(i);
  }

  // Do inverse kinematics
  for (int i = 0; i < FLAGS_n_mode; i++) {
    if (start_with_left_stance)
      left_stance = (i % 2) ? false : true;
    else
      left_stance = (i % 2) ? true : false;

    for (int j = 1; j < FLAGS_n_nodes - 1; j++) {
      MathematicalProgram math_prog;
      auto r = math_prog.NewContinuousVariables(n_r, "r");
      auto q = math_prog.NewContinuousVariables(7, "q");

      // Full order model joint limits
      cout << "Adding full-order model joint constraint...\n";
      vector<string> l_or_r{"left_", "right_"};
      vector<string> fom_joint_names{"hip_pin", "knee_pin"};
      vector<double> lb_for_fom_joints{ -M_PI / 2.0, 5.0 / 180.0 * M_PI};
      vector<double> ub_for_fom_joints{ M_PI / 2.0, M_PI / 2.0};
      for (unsigned int k = 0; k < l_or_r.size(); k++) {
        for (unsigned int l = 0; l < fom_joint_names.size(); l++) {
          math_prog.AddLinearConstraint(
            q(positions_map.at(l_or_r[k] + fom_joint_names[l])),
            lb_for_fom_joints[l], ub_for_fom_joints[l]);
        }
      }

      // Add RoM-FoM mapping constraints
      cout << "Adding RoM-FoM mapping constraint...\n";
      VectorXd r_i = states.col(j + (FLAGS_n_nodes - 1) * i).head(n_r);

      MatrixXd grad_r = MatrixXd::Zero(n_r, 7);
      AutoDiffVecXd r_i_autoDiff = initializeAutoDiff(r_i);
      drake::math::initializeAutoDiffGivenGradientMatrix(
        r_i, grad_r, r_i_autoDiff);
      // auto kin_constraint = std::make_shared<planning::KinematicsConstraintGivenR>(
      //                         n_r, r_i_autoDiff, 7, FLAGS_n_feature_kin, theta_kin);
      auto kin_constraint = std::make_shared<planning::KinematicsConstraintOnlyPos>(
                              n_r, 7, FLAGS_n_feature_kin, theta_kin);
      if (left_stance) {
        // math_prog.AddConstraint(kin_constraint, q);
        math_prog.AddConstraint(kin_constraint, {r, q});
      } else {
        VectorXDecisionVariable q_swap(7);
        q_swap << q.segment(0, 3),
               q.segment(4, 1),
               q.segment(3, 1),
               q.segment(6, 1),
               q.segment(5, 1);
        // math_prog.AddConstraint(kin_constraint, q_swap);
        math_prog.AddConstraint(kin_constraint, {r, q_swap});
        math_prog.AddLinearConstraint(r == r_i);
      }

      // Add stance foot constraint
      cout << "Adding full-order model stance foot constraint...\n";
      auto fom_sf_constraint =
        std::make_shared<planning::FomStanceFootConstraintGivenPos>(
          left_stance, n_q, stance_foot_pos_each_mode.col(i));
      math_prog.AddConstraint(fom_sf_constraint, q);

      // Add cost
      // MatrixXd Id = MatrixXd::Identity(1, 1);
      // VectorXd zero_1d_vec = VectorXd::Zero(1);
      // math_prog.AddQuadraticErrorCost(Id, zero_1d_vec, q.segment(2, 1));
      MatrixXd Id = MatrixXd::Identity(7, 7);
      VectorXd zero_7d_vec = VectorXd::Zero(7);
      VectorXd interpolated_q = x0_each_mode.col(i).head(7) +
                                (xf_each_mode.col(i).head(7) - x0_each_mode.col(i).head(7))
                                * j / (FLAGS_n_nodes - 1);
      // cout << "interpolated_q = " << interpolated_q.transpose() << endl;
      math_prog.AddQuadraticErrorCost(Id, interpolated_q, q);

      // Set initial guess
      math_prog.SetInitialGuess(r, r_i);
      math_prog.SetInitialGuess(q, interpolated_q);

      // math_prog.SetSolverOption(drake::solvers::GurobiSolver::id(), "BarConvTol", 1E-9);
      // math_prog.SetSolverOption(drake::solvers::SnoptSolver::id(),
      //                           "Print file", "snopt.out");
      // math_prog.SetSolverOption(drake::solvers::SnoptSolver::id(),
      //                           "Major iterations limit", 10000);
      // math_prog.SetSolverOption(drake::solvers::SnoptSolver::id(), "Major feasibility tolerance", 1.0e-14); //1.0e-10
      // math_prog.SetSolverOption(drake::solvers::SnoptSolver::id(), "Minor feasibility tolerance", 1.0e-14); //1.0e-10
      cout << "Start solving...\n";
      const auto result = Solve(math_prog);
      auto solution_result = result.get_solution_result();
      cout << solution_result << endl;
      cout << "Cost:" << result.get_optimal_cost() << endl;
      VectorXd q_sol = result.GetSolution(q);

      q_at_all_knots.col(j + (FLAGS_n_nodes - 1) * i) = q_sol;
      // q_at_all_knots.col(j + (FLAGS_n_nodes - 1) * i) = interpolated_q;
    }
  }


  // TODO: When you create the video, you need to adjust the ratio of duration
  // for each mode. (look at the time)
  // Right now, we can just set it to be equal time, cause you don't have time.


  writeCSV(dir_data + string("q_at_knots_IK.csv"), q_at_all_knots);


  // Create a testing piecewise polynomial
  double dt = 0.1;
  std::vector<double> T_breakpoint;
  for (int i = 0; i < states.cols(); i++)
    T_breakpoint.push_back(i * dt);
  std::vector<MatrixXd> Y;
  for (int i = 0; i < states.cols(); i++) {
    VectorXd qv(14);
    qv << q_at_all_knots.col(i), VectorXd::Zero(7);
    Y.push_back(qv);
  }
  PiecewisePolynomial<double> pp_xtraj =
    PiecewisePolynomial<double>::FirstOrderHold(T_breakpoint, Y);

  // Create MBP
  drake::systems::DiagramBuilder<double> builder;
  MultibodyPlant<double> plant;
  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  Parser parser(&plant, &scene_graph);
  std::string full_name = FindResourceOrThrow(
                            "examples/Goldilocks_models/PlanarWalkerWithTorso.urdf");
  parser.AddModelFromFile(full_name);
  plant.AddForceElement<drake::multibody::UniformGravityFieldElement>(
    -9.81 * Eigen::Vector3d::UnitZ());
  plant.WeldFrames(
    plant.world_frame(), plant.GetFrameByName("base"),
    drake::math::RigidTransform<double>());
  plant.Finalize();

  // visualizer
  multibody::connectTrajectoryVisualizer(&plant, &builder, &scene_graph,
                                         pp_xtraj);
  auto diagram = builder.Build();
  // while (true)
  drake::systems::Simulator<double> simulator(*diagram);
  simulator.set_target_realtime_rate(1);
  simulator.Initialize();
  simulator.StepTo(pp_xtraj.end_time());

  return;
}
}  // namespace planning
}  // namespace goldilocks_models
}  // namespace dairlib

int main(int argc, char* argv[]) {
  dairlib::goldilocks_models::planning::visualizeFullOrderModelTraj(argc, argv);
  return 0;
}


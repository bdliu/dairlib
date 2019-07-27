//TODO(yminchen): finish this file

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

#include "drake/solvers/solve.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/constraint.h"
#include "solvers/optimization_utils.h"
#include "systems/trajectory_optimization/dircon_position_data.h"
#include "systems/trajectory_optimization/dircon_kinematic_data_set.h"
#include "systems/trajectory_optimization/dircon_opt_constraints.h"
#include "systems/trajectory_optimization/hybrid_dircon.h"

#include "multibody/multibody_utils.h"
#include "multibody/visualization_utils.h"

#include "systems/goldilocks_models/symbolic_manifold.h"
#include "systems/goldilocks_models/file_utils.h"

using std::vector;
using std::shared_ptr;
using std::cout;
using std::endl;
using std::string;
using std::map;

using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Matrix3Xd;

using drake::VectorX;
using drake::systems::trajectory_optimization::MultipleShooting;
using drake::solvers::Binding;
using drake::solvers::Constraint;
using drake::solvers::VectorXDecisionVariable;
using drake::solvers::MatrixXDecisionVariable;
using drake::solvers::SolutionResult;
using drake::symbolic::Variable;
using drake::symbolic::Expression;
using drake::trajectories::PiecewisePolynomial;

using drake::multibody::MultibodyPlant;
using drake::geometry::SceneGraph;
using drake::multibody::Body;
using drake::multibody::Parser;
using drake::systems::rendering::MultibodyPositionToGeometryPose;

using drake::multibody::JointActuator;
using drake::multibody::JointActuatorIndex;

namespace dairlib {

using systems::trajectory_optimization::HybridDircon;
using systems::trajectory_optimization::DirconDynamicConstraint;
using systems::trajectory_optimization::DirconKinematicConstraint;
using systems::trajectory_optimization::DirconOptions;
using systems::trajectory_optimization::DirconKinConstraintType;
using systems::SubvectorPassThrough;

using goldilocks_models::readCSV;
using goldilocks_models::writeCSV;

/// Currently, MBP doesn't support close-loop linkage, so we add distance
/// constraints in trajectory optimization (dircon).

/// This file runs trajectory optimization for fixed-spring cassie

void DoMain(double stride_length, double duration, int iter,
            string data_directory,
            string init_file,
            string output_prefix) {

  drake::systems::DiagramBuilder<double> builder;
  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  scene_graph.set_name("scene_graph");

  MultibodyPlant<double> plant;
  multibody::addFlatTerrain(&plant, &scene_graph, .8, .8);
  Parser parser(&plant, &scene_graph);

  std::string full_name =
    FindResourceOrThrow("examples/Cassie/urdf/cassie_fixed_springs.urdf");
  parser.AddModelFromFile(full_name);
  plant.mutable_gravity_field().set_gravity_vector(
    -9.81 * Eigen::Vector3d::UnitZ());
  plant.Finalize();

  // Create maps for joints
  map<string, int> positions_map = multibody::makeNameToPositionsMap(plant);
  map<string, int> velocities_map = multibody::makeNameToVelocitiesMap(plant);
  map<string, int> actuators_map = multibody::makeNameToActuatorsMap(plant);
  // for (auto const& element : positions_map)
  //   cout << element.first << " = " << element.second << endl;
  // cout << "\n";
  // for (auto const& element : velocities_map)
  //   cout << element.first << " = " << element.second << endl;
  // cout << "\n";
  // for (auto const& element : actuators_map)
  //   cout << element.first << " = " << element.second << endl;
  // cout << "\n";

  // ankle_joint_left = 15
  // ankle_joint_right = 16
  // hip_pitch_left = 11
  // hip_pitch_right = 12
  // hip_roll_left = 7
  // hip_roll_right = 8
  // hip_yaw_left = 9
  // hip_yaw_right = 10
  // knee_left = 13
  // knee_right = 14
  // position[0] = 0
  // position[1] = 1
  // position[2] = 2
  // position[3] = 3
  // position[4] = 4
  // position[5] = 5
  // position[6] = 6
  // toe_left = 17
  // toe_right = 18

  // ankle_joint_leftdot = 14
  // ankle_joint_rightdot = 15
  // hip_pitch_leftdot = 10
  // hip_pitch_rightdot = 11
  // hip_roll_leftdot = 6
  // hip_roll_rightdot = 7
  // hip_yaw_leftdot = 8
  // hip_yaw_rightdot = 9
  // knee_leftdot = 12
  // knee_rightdot = 13
  // toe_leftdot = 16
  // toe_rightdot = 17
  // velocity[0] = 0
  // velocity[1] = 1
  // velocity[2] = 2
  // velocity[3] = 3
  // velocity[4] = 4
  // velocity[5] = 5

  // hip_pitch_left_motor = 4
  // hip_pitch_right_motor = 5
  // hip_roll_left_motor = 0
  // hip_roll_right_motor = 1
  // hip_yaw_left_motor = 2
  // hip_yaw_right_motor = 3
  // knee_left_motor = 6
  // knee_right_motor = 7
  // toe_left_motor = 8
  // toe_right_motor = 9

  int n_q = plant.num_positions();
  int n_v = plant.num_velocities();
  int n_u = plant.num_actuators();
  // int n_x = n_q + n_v;
  // cout<<"n_x = "<<n_x<<"\n";
  // cout<<"n_u = "<<n_u<<"\n";

  // Set up contact constraints and construct dircon
  const Body<double>& toe_left = plant.GetBodyByName("toe_left");
  const Body<double>& toe_right = plant.GetBodyByName("toe_right");
  Vector3d front_contact_disp(-0.0457, 0.112, 0);
  Vector3d rear_contact_disp(0.088, 0, 0);
  bool isXZ = false;
  auto left_toe_front_constraint = DirconPositionData<double>(plant, toe_left,
                                   front_contact_disp, isXZ);
  auto left_toe_rear_constraint = DirconPositionData<double>(plant, toe_left,
                                  rear_contact_disp, isXZ);
  auto right_toe_front_constraint = DirconPositionData<double>(plant, toe_right,
                                    front_contact_disp, isXZ);
  auto right_toe_rear_constraint = DirconPositionData<double>(plant, toe_right,
                                   rear_contact_disp, isXZ);

  Vector3d normal;
  normal << 0, 0, 1;
  double mu = 1;
  left_toe_front_constraint.addFixedNormalFrictionConstraints(normal, mu);
  left_toe_rear_constraint.addFixedNormalFrictionConstraints(normal, mu);
  right_toe_front_constraint.addFixedNormalFrictionConstraints(normal, mu);
  right_toe_rear_constraint.addFixedNormalFrictionConstraints(normal, mu);

  std::vector<DirconKinematicData<double>*> left_toe_constraint;
  left_toe_constraint.push_back(&left_toe_front_constraint);
  left_toe_constraint.push_back(&left_toe_rear_constraint);
  auto left_data_set = DirconKinematicDataSet<double>(plant,
                       &left_toe_constraint);

  std::vector<DirconKinematicData<double>*> right_toe_constraint;
  right_toe_constraint.push_back(&right_toe_front_constraint);
  right_toe_constraint.push_back(&right_toe_rear_constraint);
  auto right_data_set = DirconKinematicDataSet<double>(plant,
                        &right_toe_constraint);

  auto left_options = DirconOptions(left_data_set.countConstraints());
  left_options.setConstraintRelative(0, true);
  left_options.setConstraintRelative(1, true);
  left_options.setConstraintRelative(3, true);
  left_options.setConstraintRelative(4, true);
  auto right_options = DirconOptions(right_data_set.countConstraints());
  right_options.setConstraintRelative(0, true);
  right_options.setConstraintRelative(1, true);
  right_options.setConstraintRelative(3, true);
  right_options.setConstraintRelative(4, true);

  // Stated in the MultipleShooting class:
  // This class assumes that there are a fixed number (N) time steps/samples,
  // and that the trajectory is discretized into timesteps h (N-1 of these),
  // state x (N of these), and control input u (N of these).
  std::vector<int> num_time_samples;
  num_time_samples.push_back(20); // First mode (20 sample points)
  num_time_samples.push_back(1);  // Second mode (1 sample point)
  std::vector<double> min_dt;
  min_dt.push_back(.01);
  min_dt.push_back(.01);
  std::vector<double> max_dt;
  max_dt.push_back(.3);
  max_dt.push_back(.3);

  int N = 0;
  for (uint i = 0; i < num_time_samples.size(); i++)
    N += num_time_samples[i];
  N -= num_time_samples.size() - 1;  // because of overlaps between modes

  std::vector<DirconKinematicDataSet<double>*> dataset_list;
  dataset_list.push_back(&left_data_set);
  dataset_list.push_back(&right_data_set);

  std::vector<DirconOptions> options_list;
  options_list.push_back(left_options);
  options_list.push_back(right_options);

  auto trajopt = std::make_shared<HybridDircon<double>>(plant,
                 num_time_samples, min_dt, max_dt, dataset_list, options_list);

  // trajopt->AddDurationBounds(duration, duration);

  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Print file", "snopt.out");
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Major iterations limit", iter);
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(), "Verify level", 0);

  // Periodicity constraints
  auto x0 = trajopt->initial_state();
  auto xf = trajopt->state_vars_by_mode(num_time_samples.size() - 1,
                                        num_time_samples[num_time_samples.size() - 1] - 1);
  // Floating base (z and rotation) should be the same
  trajopt->AddLinearConstraint(x0(positions_map.at("position[2]")) == xf(
                                 positions_map.at("position[2]")));
  trajopt->AddLinearConstraint(x0(positions_map.at("position[3]")) == xf(
                                 positions_map.at("position[3]")));
  trajopt->AddLinearConstraint(x0(positions_map.at("position[4]")) == xf(
                                 positions_map.at("position[4]")));
  trajopt->AddLinearConstraint(x0(positions_map.at("position[5]")) == xf(
                                 positions_map.at("position[5]")));
  trajopt->AddLinearConstraint(x0(positions_map.at("position[6]")) == xf(
                                 positions_map.at("position[6]")));
  trajopt->AddLinearConstraint(
    x0(n_q + velocities_map.at("velocity[0]")) ==
    xf(n_q + velocities_map.at("velocity[0]")));
  trajopt->AddLinearConstraint(
    x0(n_q + velocities_map.at("velocity[2]")) ==
    xf(n_q + velocities_map.at("velocity[2]")));
  // TODO: Not sure how to impose period constraint in rotation for 3D walking

  // The legs joint positions and velocities should be mirrored between legs
  trajopt->AddLinearConstraint(x0(positions_map.at("ankle_joint_left")) == xf(
                                 positions_map.at("ankle_joint_right")));
  trajopt->AddLinearConstraint(x0(positions_map.at("hip_pitch_left")) == xf(
                                 positions_map.at("hip_pitch_right")));
  trajopt->AddLinearConstraint(x0(positions_map.at("hip_roll_left")) == xf(
                                 positions_map.at("hip_roll_right")));
  trajopt->AddLinearConstraint(x0(positions_map.at("hip_yaw_left")) == xf(
                                 positions_map.at("hip_yaw_right")));
  trajopt->AddLinearConstraint(x0(positions_map.at("knee_left")) == xf(
                                 positions_map.at("knee_right")));
  trajopt->AddLinearConstraint(x0(positions_map.at("toe_left")) == xf(
                                 positions_map.at("toe_right")));
  trajopt->AddLinearConstraint(x0(positions_map.at("ankle_joint_right")) == xf(
                                 positions_map.at("ankle_joint_left")));
  trajopt->AddLinearConstraint(x0(positions_map.at("hip_pitch_right")) == xf(
                                 positions_map.at("hip_pitch_left")));
  trajopt->AddLinearConstraint(x0(positions_map.at("hip_roll_right")) == xf(
                                 positions_map.at("hip_roll_left")));
  trajopt->AddLinearConstraint(x0(positions_map.at("hip_yaw_right")) == xf(
                                 positions_map.at("hip_yaw_left")));
  trajopt->AddLinearConstraint(x0(positions_map.at("knee_right")) == xf(
                                 positions_map.at("knee_left")));
  trajopt->AddLinearConstraint(x0(positions_map.at("toe_right")) == xf(
                                 positions_map.at("toe_left")));
  trajopt->AddLinearConstraint(
    x0(n_q + velocities_map.at("ankle_joint_leftdot")) ==
    xf(n_q + velocities_map.at("ankle_joint_rightdot")));
  trajopt->AddLinearConstraint(
    x0(n_q + velocities_map.at("hip_pitch_leftdot")) ==
    xf(n_q + velocities_map.at("hip_pitch_rightdot")));
  trajopt->AddLinearConstraint(
    x0(n_q + velocities_map.at("hip_roll_leftdot")) ==
    xf(n_q + velocities_map.at("hip_roll_rightdot")));
  trajopt->AddLinearConstraint(
    x0(n_q + velocities_map.at("hip_yaw_leftdot")) ==
    xf(n_q + velocities_map.at("hip_yaw_rightdot")));
  trajopt->AddLinearConstraint(
    x0(n_q + velocities_map.at("knee_leftdot")) ==
    xf(n_q + velocities_map.at("knee_rightdot")));
  trajopt->AddLinearConstraint(
    x0(n_q + velocities_map.at("toe_leftdot")) ==
    xf(n_q + velocities_map.at("toe_rightdot")));
  trajopt->AddLinearConstraint(
    x0(n_q + velocities_map.at("ankle_joint_rightdot")) ==
    xf(n_q + velocities_map.at("ankle_joint_leftdot")));
  trajopt->AddLinearConstraint(
    x0(n_q + velocities_map.at("hip_pitch_rightdot")) ==
    xf(n_q + velocities_map.at("hip_pitch_leftdot")));
  trajopt->AddLinearConstraint(
    x0(n_q + velocities_map.at("hip_roll_rightdot")) ==
    xf(n_q + velocities_map.at("hip_roll_leftdot")));
  trajopt->AddLinearConstraint(
    x0(n_q + velocities_map.at("hip_yaw_rightdot")) ==
    xf(n_q + velocities_map.at("hip_yaw_leftdot")));
  trajopt->AddLinearConstraint(
    x0(n_q + velocities_map.at("knee_rightdot")) ==
    xf(n_q + velocities_map.at("knee_leftdot")));
  trajopt->AddLinearConstraint(
    x0(n_q + velocities_map.at("toe_rightdot")) ==
    xf(n_q + velocities_map.at("toe_leftdot")));

  // u periodic constraint
  auto u0 = trajopt->input(0);
  auto uf = trajopt->input(N - 1);
  std::vector<std::string> left_motor_names {
    "hip_pitch_left_motor",
    "hip_roll_left_motor",
    "hip_yaw_left_motor",
    "knee_left_motor",
    "toe_left_motor",
  };
  std::vector<std::string> right_motor_names {
    "hip_pitch_right_motor",
    "hip_roll_right_motor",
    "hip_yaw_right_motor",
    "knee_right_motor",
    "toe_right_motor"
  };
  for (int i = 0; i < left_motor_names.size(); i++) {
    trajopt->AddLinearConstraint(u0(actuators_map.at(left_motor_names[i])) ==
                                 uf(actuators_map.at(right_motor_names[i])));
    trajopt->AddLinearConstraint(u0(actuators_map.at(right_motor_names[i])) ==
                                 uf(actuators_map.at(left_motor_names[i])));
  }

  // u limit
  auto u = trajopt->input();
  std::vector<std::string> motor_names {};
  motor_names.insert(motor_names.end(),
                     left_motor_names.begin(), left_motor_names.end() );
  motor_names.insert(motor_names.end(),
                     right_motor_names.begin(), right_motor_names.end() );
  for (const auto & member : motor_names) {
    trajopt->AddConstraintToAllKnotPoints(u(actuators_map.at(member)) <= 300);
    trajopt->AddConstraintToAllKnotPoints(u(actuators_map.at(member)) >= -300);
  }

  // joint limits
  auto x = trajopt->state();
  std::vector<std::string> leg_pos_joint_names {
    "hip_roll_left",
    "hip_roll_right",
    "hip_yaw_left",
    "hip_yaw_right",
    "hip_pitch_left",
    "hip_pitch_right",
    "knee_left",
    "knee_right",
    "ankle_joint_left",
    "ankle_joint_right",
    "toe_left",
    "toe_right"};
  for (const auto & member : leg_pos_joint_names) {
    trajopt->AddConstraintToAllKnotPoints(
      x(positions_map.at(member)) <=
      plant.GetJointByName(member).position_upper_limits()(0));
    trajopt->AddConstraintToAllKnotPoints(
      x(positions_map.at(member)) >=
      plant.GetJointByName(member).position_lower_limits()(0));
  }

  // x-distance constraint constraints
  trajopt->AddLinearConstraint(x0(positions_map.at("position[0]")) == 0);
  trajopt->AddLinearConstraint(xf(positions_map.at("position[0]")) ==
                               stride_length);

  // make sure it's left stance
  // trajopt->AddLinearConstraint(x0(positions_map.at("left_hip_pin")) <=
  //                              x0(positions_map.at("right_hip_pin")));

  // add cost
  const double R = 10;  // Cost on input effort
  trajopt->AddRunningCost(u.transpose()*R * u);
  MatrixXd Q = MatrixXd::Zero(2 * n_q, 2 * n_q);
  for (int i = 0; i < n_q; i++) {
    Q(i + n_q, i + n_q) = 10;
  }
  trajopt->AddRunningCost(x.transpose()*Q * x);

  // initial guess if the file exists
  if (!init_file.empty()) {
    MatrixXd z0 = readCSV(data_directory + init_file);
    trajopt->SetInitialGuessForAllVariables(z0);
  }

  cout << "Solving DIRCON (based on MultipleShooting)\n";
  auto start = std::chrono::high_resolution_clock::now();
  const auto result = Solve(*trajopt, trajopt->initial_guess());
  SolutionResult solution_result = result.get_solution_result();
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  // trajopt->PrintSolution();
  cout << to_string(solution_result) << endl;
  cout << "Solve time:" << elapsed.count() << std::endl;
  cout << "Cost:" << result.get_optimal_cost() << std::endl;

  // store the solution of the decision variable
  VectorXd z = result.GetSolution(trajopt->decision_variables());
  writeCSV(data_directory + output_prefix + string("z.csv"), z);

  // store the time, state, and input at knot points
  // VectorXd time_at_knot_point = trajopt->GetSampleTimes();
  // MatrixXd state_at_knot_point = trajopt->GetStateSamples();
  // MatrixXd input_at_knot_point = trajopt->GetInputSamples();
  // writeCSV(data_directory + string("t_i.csv"), time_at_knot_point);
  // writeCSV(data_directory + string("x_i.csv"), state_at_knot_point);
  // writeCSV(data_directory + string("u_i.csv"), input_at_knot_point);
  // cout<<"time_at_knot_point = \n"<<time_at_knot_point<<"\n";
  // cout<<state_at_knot_point.rows()<<", "<<state_at_knot_point.cols()<<"\n";
  // cout<<"state_at_knot_point = \n"<<state_at_knot_point<<"\n";
  // cout<<input_at_knot_point.rows()<<", "<<input_at_knot_point.cols()<<"\n";
  // cout<<"input_at_knot_point = \n"<<input_at_knot_point<<"\n";

  // visualizer
  const PiecewisePolynomial<double> pp_xtraj =
    trajopt->ReconstructStateTrajectory(result);
  multibody::connectTrajectoryVisualizer(&plant, &builder, &scene_graph,
                                         pp_xtraj);

  auto diagram = builder.Build();

  while (true) {
    drake::systems::Simulator<double> simulator(*diagram);
    simulator.set_target_realtime_rate(.1);
    simulator.Initialize();
    simulator.StepTo(pp_xtraj.end_time());
  }

  return ;
}

}  // namespace dairlib


int main() {
  double stride_length = 0.4;
  double duration = .5;
  int iter = 500;
  string data_directory = "examples/Cassie/trajopt_data/";
  string init_file = "";
  // string init_file = "testing_z.csv";
  string output_prefix = "testing_";

  dairlib::DoMain(stride_length, duration, iter,
                  data_directory, init_file, output_prefix);
}


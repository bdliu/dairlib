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

#include "drake/solvers/solve.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/constraint.h"

#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"

#include "common/find_resource.h"
#include "systems/primitives/subvector_pass_through.h"

#include "solvers/optimization_utils.h"
#include "systems/trajectory_optimization/dircon_position_data.h"
#include "systems/trajectory_optimization/dircon_distance_data.h"
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

using drake::math::RotationMatrix;
using drake::math::RollPitchYaw;

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


// Do inverse kinematics to get configuration guess
vector<VectorXd> GetInitGuessForQ(int N,
                                  double stride_length,
                                  const MultibodyPlant<double>& plant) {
  int n_q = plant.num_positions();
  int n_v = plant.num_velocities();
  int n_x = n_q + n_v;
  map<string, int> positions_map = multibody::makeNameToPositionsMap(plant);

  vector<VectorXd> q_init_guess;
  VectorXd q_ik_guess = VectorXd::Zero(n_q);
  Eigen::Vector4d quat(2000.06,
                       -0.339462,
                       -0.609533,
                       -0.760854);
  q_ik_guess << quat.normalized(),
             0.000889849,
             0.000626865,
             1.0009,
             -0.0112109,
             0.00927845,
             -0.000600725,
             -0.000895805,
             1.15086,
             0.610808,
             -1.38608,
             -1.35926,
             0.806192,
             1.00716,
             -M_PI / 2,
             -M_PI / 2;

  for (int i = 0; i < N; i++) {
    double eps = 1e-3;
    Vector3d eps_vec = eps * VectorXd::Ones(3);
    Vector3d pelvis_pos(stride_length * i / (N - 1),
                        0.0,
                        1.0);
    Vector3d stance_toe_pos(stride_length / 2,
                            0.12,
                            0.05);
    Vector3d swing_toe_pos(-stride_length / 2 + 2 * stride_length * i / (N - 1),
                           -0.12,
                           0.05 + 0.1 * (-abs((i - N / 2.0) / (N / 2.0)) + 1));
    // cout << "swing foot height = " <<
    //      0.05 + 0.1 * (-abs((i - N / 2.0) / (N / 2.0)) + 1);

    const auto & world_frame = plant.world_frame();
    const auto & pelvis_frame = plant.GetFrameByName("pelvis");
    const auto & toe_left_frame = plant.GetFrameByName("toe_left");
    const auto & toe_right_frame = plant.GetFrameByName("toe_right");

    drake::multibody::InverseKinematics ik(plant);
    ik.AddPositionConstraint(pelvis_frame, Vector3d(0, 0, 0),
                             world_frame,
                             pelvis_pos - eps * VectorXd::Ones(3),
                             pelvis_pos + eps * VectorXd::Ones(3));
    ik.AddOrientationConstraint(pelvis_frame, RotationMatrix<double>(),
                                world_frame, RotationMatrix<double>(),
                                eps);
    ik.AddPositionConstraint(toe_left_frame, Vector3d(0, 0, 0),
                             world_frame,
                             stance_toe_pos - eps_vec,
                             stance_toe_pos + eps_vec);
    ik.AddPositionConstraint(toe_right_frame, Vector3d(0, 0, 0),
                             world_frame,
                             swing_toe_pos - eps_vec,
                             swing_toe_pos + eps_vec);
    ik.get_mutable_prog()->AddLinearConstraint(
      (ik.q())(positions_map.at("hip_yaw_left")) == 0);
    ik.get_mutable_prog()->AddLinearConstraint(
      (ik.q())(positions_map.at("hip_yaw_right")) == 0);
    // Four bar linkage constraint (without spring)
    ik.get_mutable_prog()->AddLinearConstraint(
      (ik.q())(positions_map.at("knee_left"))
      + (ik.q())(positions_map.at("ankle_joint_left")) == M_PI * 13 / 180.0);
    ik.get_mutable_prog()->AddLinearConstraint(
      (ik.q())(positions_map.at("knee_right"))
      + (ik.q())(positions_map.at("ankle_joint_right")) == M_PI * 13 / 180.0);

    ik.get_mutable_prog()->SetInitialGuess(ik.q(),
                                           q_ik_guess);
    const auto result = Solve(ik.prog());
    SolutionResult solution_result = result.get_solution_result();
    cout << "\n" << to_string(solution_result) << endl;
    cout << "  Cost:" << result.get_optimal_cost() << std::endl;
    const auto q_sol = result.GetSolution(ik.q());
    cout << "  q_sol" << q_sol.transpose() << endl;
    VectorXd q_sol_normd(n_q);
    q_sol_normd << q_sol.head(4).normalized(), q_sol.tail(n_q - 4);
    q_ik_guess = q_sol_normd;
    q_init_guess.push_back(q_sol_normd);

    // Build temporary diagram for visualization
    drake::systems::DiagramBuilder<double> builder_ik;
    SceneGraph<double>& scene_graph_ik = *builder_ik.AddSystem<SceneGraph>();
    scene_graph_ik.set_name("scene_graph_ik");
    MultibodyPlant<double> plant_ik;
    multibody::addFlatTerrain(&plant_ik, &scene_graph_ik, .8, .8);
    Parser parser(&plant_ik, &scene_graph_ik);
    std::string full_name =
      FindResourceOrThrow("examples/Cassie/urdf/cassie_fixed_springs.urdf");
    parser.AddModelFromFile(full_name);
    plant_ik.mutable_gravity_field().set_gravity_vector(
      -9.81 * Eigen::Vector3d::UnitZ());
    plant_ik.Finalize();

    // Visualize
    VectorXd x_const = VectorXd::Zero(n_x);
    x_const.head(n_q) = q_sol;
    PiecewisePolynomial<double> pp_xtraj(x_const);

    multibody::connectTrajectoryVisualizer(&plant_ik, &builder_ik,
                                           &scene_graph_ik, pp_xtraj);
    auto diagram = builder_ik.Build();
    drake::systems::Simulator<double> simulator(*diagram);
    simulator.set_target_realtime_rate(.1);
    simulator.Initialize();
    simulator.StepTo(1.0 / N);
  }

  return q_init_guess;
}

// Get v by finite differencing q
vector<VectorXd> GetInitGuessForV(const vector<VectorXd>& q_seed, double dt,
                                  const MultibodyPlant<double>& plant) {
  vector<VectorXd> qdot_seed;
  for (unsigned int i = 0; i < q_seed.size(); i++) {
    if (i == 0) {
      qdot_seed.push_back((q_seed[i + 1] - q_seed[i]) / dt);
    } else {
      qdot_seed.push_back((q_seed[i] - q_seed[i - 1]) / dt);
    }
  }

  // Convert qdot to v
  vector<VectorXd> v_seed;
  for (unsigned int i = 0; i < q_seed.size(); i++) {
    auto context = plant.CreateDefaultContext();
    plant.SetPositions(context.get(), q_seed[i]);
    VectorXd v(plant.num_velocities());
    plant.MapQDotToVelocity(*context, qdot_seed[i], &v);
    v_seed.push_back(v);
    cout << i << ":\n";
    cout << "  qdot = " << qdot_seed[i].transpose() << endl;
    cout << "  v = " << v.transpose() << endl;
  }
  return v_seed;
}

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

  // position[0] = 0
  // position[1] = 1
  // position[2] = 2
  // position[3] = 3
  // position[4] = 4
  // position[5] = 5
  // position[6] = 6
  // hip_roll_left = 7
  // hip_roll_right = 8
  // hip_yaw_left = 9
  // hip_yaw_right = 10
  // hip_pitch_left = 11
  // hip_pitch_right = 12
  // knee_left = 13
  // knee_right = 14
  // ankle_joint_left = 15
  // ankle_joint_right = 16
  // toe_left = 17
  // toe_right = 18

  // velocity[0] = 0
  // velocity[1] = 1
  // velocity[2] = 2
  // velocity[3] = 3
  // velocity[4] = 4
  // velocity[5] = 5
  // hip_roll_leftdot = 6
  // hip_roll_rightdot = 7
  // hip_yaw_leftdot = 8
  // hip_yaw_rightdot = 9
  // hip_pitch_leftdot = 10
  // hip_pitch_rightdot = 11
  // knee_leftdot = 12
  // knee_rightdot = 13
  // ankle_joint_leftdot = 14
  // ankle_joint_rightdot = 15
  // toe_leftdot = 16
  // toe_rightdot = 17

  // hip_roll_left_motor = 0
  // hip_roll_right_motor = 1
  // hip_yaw_left_motor = 2
  // hip_yaw_right_motor = 3
  // hip_pitch_left_motor = 4
  // hip_pitch_right_motor = 5
  // knee_left_motor = 6
  // knee_right_motor = 7
  // toe_left_motor = 8
  // toe_right_motor = 9

  int n_q = plant.num_positions();
  int n_v = plant.num_velocities();
  int n_u = plant.num_actuators();
  // int n_x = n_q + n_v;
  cout << "n_q = " << n_q << "\n";
  cout << "n_v = " << n_v << "\n";
  // cout << "n_x = " << n_x << "\n";
  // cout << "n_u = " << n_u << "\n";
  cout << "floating_positions_start = " <<
       plant.GetBodyByName("pelvis").floating_positions_start() << endl;
  cout << "floating_velocities_start = " <<
       plant.GetBodyByName("pelvis").floating_velocities_start() << endl;

  // Set up contact/distance constraints and construct dircon
  const Body<double>& toe_left = plant.GetBodyByName("toe_left");
  const Body<double>& toe_right = plant.GetBodyByName("toe_right");
  Vector3d pt_front_contact(-0.0457, 0.112, 0);
  Vector3d pt_rear_contact(0.088, 0, 0);
  bool isXZ = false;
  auto left_toe_front_constraint = DirconPositionData<double>(plant, toe_left,
                                   pt_front_contact, isXZ);
  auto left_toe_rear_constraint = DirconPositionData<double>(plant, toe_left,
                                  pt_rear_contact, isXZ);
  auto right_toe_front_constraint = DirconPositionData<double>(plant, toe_right,
                                    pt_front_contact, isXZ);
  auto right_toe_rear_constraint = DirconPositionData<double>(plant, toe_right,
                                   pt_rear_contact, isXZ);
  Vector3d normal(0, 0, 1);
  double mu = 1;
  left_toe_front_constraint.addFixedNormalFrictionConstraints(normal, mu);
  left_toe_rear_constraint.addFixedNormalFrictionConstraints(normal, mu);
  right_toe_front_constraint.addFixedNormalFrictionConstraints(normal, mu);
  right_toe_rear_constraint.addFixedNormalFrictionConstraints(normal, mu);

  const auto & thigh_left = plant.GetBodyByName("thigh_left");
  const auto & heel_spring_left = plant.GetBodyByName("heel_spring_left");
  const auto & thigh_right = plant.GetBodyByName("thigh_right");
  const auto & heel_spring_right = plant.GetBodyByName("heel_spring_right");
  double rod_length = 0.5012;  // from cassie_utils
  Vector3d pt_on_heel_spring = Vector3d(.11877, -.01, 0.0);
  Vector3d pt_on_thigh_left = Vector3d(0.0, 0.0, 0.045);
  Vector3d pt_on_thigh_right = Vector3d(0.0, 0.0, -0.045);
  auto distance_constraint_left = DirconDistanceData<double>(plant,
                                  thigh_left, pt_on_thigh_left,
                                  heel_spring_left, pt_on_heel_spring,
                                  rod_length);
  auto distance_constraint_right = DirconDistanceData<double>(plant,
                                   thigh_right, pt_on_thigh_right,
                                   heel_spring_right, pt_on_heel_spring,
                                   rod_length);

  // Testing
  Vector3d pt_mid_contact = pt_front_contact + pt_rear_contact;
  auto left_toe_mid_constraint = DirconPositionData<double>(plant, toe_left,
                                 pt_mid_contact, isXZ);
  auto right_toe_mid_constraint = DirconPositionData<double>(plant, toe_right,
                                  pt_mid_contact, isXZ);
  left_toe_mid_constraint.addFixedNormalFrictionConstraints(normal, mu);
  right_toe_mid_constraint.addFixedNormalFrictionConstraints(normal, mu);


  vector<DirconKinematicData<double>*> left_stance_constraint;
  // left_stance_constraint.push_back(&left_toe_front_constraint);
  // left_stance_constraint.push_back(&left_toe_rear_constraint);
  left_stance_constraint.push_back(&left_toe_mid_constraint);
  // left_stance_constraint.push_back(&distance_constraint_left);
  // left_stance_constraint.push_back(&distance_constraint_right);
  auto left_dataset = DirconKinematicDataSet<double>(plant,
                      &left_stance_constraint);

  vector<DirconKinematicData<double>*> right_stance_constraint;
  // right_stance_constraint.push_back(&right_toe_front_constraint);
  // right_stance_constraint.push_back(&right_toe_rear_constraint);
  right_stance_constraint.push_back(&right_toe_mid_constraint);
  // right_stance_constraint.push_back(&distance_constraint_left);
  // right_stance_constraint.push_back(&distance_constraint_right);
  auto right_dataset = DirconKinematicDataSet<double>(plant,
                       &right_stance_constraint);


  auto left_options = DirconOptions(left_dataset.countConstraints());
  left_options.setConstraintRelative(0, true);
  left_options.setConstraintRelative(1, true);
  // left_options.setConstraintRelative(3, true);
  // left_options.setConstraintRelative(4, true);
  auto right_options = DirconOptions(right_dataset.countConstraints());
  right_options.setConstraintRelative(0, true);
  right_options.setConstraintRelative(1, true);
  // right_options.setConstraintRelative(3, true);
  // right_options.setConstraintRelative(4, true);

  // Stated in the MultipleShooting class:
  // This class assumes that there are a fixed number (N) time steps/samples,
  // and that the trajectory is discretized into timesteps h (N-1 of these),
  // state x (N of these), and control input u (N of these).
  vector<int> num_time_samples;
  num_time_samples.push_back(20); // First mode (20 sample points)
  num_time_samples.push_back(1);  // Second mode (1 sample point)
  vector<double> min_dt;
  min_dt.push_back(.01);
  min_dt.push_back(.01);
  vector<double> max_dt;
  max_dt.push_back(.3);
  max_dt.push_back(.3);

  int N = 0;
  for (uint i = 0; i < num_time_samples.size(); i++)
    N += num_time_samples[i];
  N -= num_time_samples.size() - 1;  // because of overlaps between modes

  vector<DirconKinematicDataSet<double>*> dataset_list;
  dataset_list.push_back(&left_dataset);
  dataset_list.push_back(&right_dataset);

  vector<DirconOptions> options_list;
  options_list.push_back(left_options);
  options_list.push_back(right_options);

  auto trajopt = std::make_shared<HybridDircon<double>>(plant,
                 num_time_samples, min_dt, max_dt, dataset_list, options_list);

  // Fix the time duration
  trajopt->AddDurationBounds(duration, duration);

  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Print file", "snopt.out");
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Major iterations limit", iter);
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Verify level", 0);

  // Get the decision varaibles that will be used
  auto u = trajopt->input();
  auto x = trajopt->state();
  auto u0 = trajopt->input(0);
  auto uf = trajopt->input(N - 1);
  auto x0 = trajopt->initial_state();
  auto xf = trajopt->state_vars_by_mode(num_time_samples.size() - 1,
                                        num_time_samples[num_time_samples.size() - 1] - 1);

  // Initial quaterion norm constraint (set it to identity for now)
  trajopt->AddLinearConstraint(x0(positions_map.at("position[0]")) == 1);
  trajopt->AddLinearConstraint(x0(positions_map.at("position[1]")) == 0);
  trajopt->AddLinearConstraint(x0(positions_map.at("position[2]")) == 0);
  trajopt->AddLinearConstraint(x0(positions_map.at("position[3]")) == 0);


  // x-distance constraint constraints
  trajopt->AddLinearConstraint(x0(positions_map.at("position[4]")) == 0);
  trajopt->AddLinearConstraint(xf(positions_map.at("position[4]")) ==
                               stride_length);
  /*
    // Periodicity constraints
    // Floating base (z and rotation) should be the same
    trajopt->AddLinearConstraint(x0(positions_map.at("position[0]")) == xf(
                                   positions_map.at("position[0]")));
    trajopt->AddLinearConstraint(x0(positions_map.at("position[1]")) == xf(
                                   positions_map.at("position[1]")));
    trajopt->AddLinearConstraint(x0(positions_map.at("position[2]")) == xf(
                                   positions_map.at("position[2]")));
    trajopt->AddLinearConstraint(x0(positions_map.at("position[3]")) == xf(
                                   positions_map.at("position[3]")));
    trajopt->AddLinearConstraint(x0(positions_map.at("position[6]")) == xf(
                                   positions_map.at("position[6]")));
    trajopt->AddLinearConstraint(
      x0(n_q + velocities_map.at("velocity[4]")) ==
      xf(n_q + velocities_map.at("velocity[4]")));
    trajopt->AddLinearConstraint(
      x0(n_q + velocities_map.at("velocity[6]")) ==
      xf(n_q + velocities_map.at("velocity[6]")));
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
    vector<std::string> left_motor_names {
      "hip_pitch_left_motor",
      "hip_roll_left_motor",
      "hip_yaw_left_motor",
      "knee_left_motor",
      "toe_left_motor",
    };
    vector<std::string> right_motor_names {
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
    vector<std::string> motor_names {};
    motor_names.insert(motor_names.end(),
                       left_motor_names.begin(), left_motor_names.end() );
    motor_names.insert(motor_names.end(),
                       right_motor_names.begin(), right_motor_names.end() );
    for (const auto & member : motor_names) {
      trajopt->AddConstraintToAllKnotPoints(u(actuators_map.at(member)) <= 300);
      trajopt->AddConstraintToAllKnotPoints(u(actuators_map.at(member)) >= -300);
    }

    // joint limits
    vector<std::string> leg_pos_joint_names {
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
    }*/

  // make sure it's left stance
  // trajopt->AddLinearConstraint(x0(positions_map.at("left_hip_pin")) <=
  //                              x0(positions_map.at("right_hip_pin")));









  // // Testing
  // int total_rows = 0;
  // auto constraint_binding_vec = trajopt->GetAllConstraints();
  // for (int i = 0; i < constraint_binding_vec.size(); i++) {
  //   const auto & binding = constraint_binding_vec[i];
  //   cout << "Constraint row " << total_rows << " to row " <<
  //        total_rows + binding.evaluator()->num_constraints() << ". Vars: " <<
  //        binding.variables().transpose() << endl;
  //   total_rows += binding.evaluator()->num_constraints();
  // }





  // add cost
  const double R = 10;  // Cost on input effort
  trajopt->AddRunningCost(u.transpose()*R * u);
  MatrixXd Q = MatrixXd::Zero(2 * n_q, 2 * n_q);
  for (int i = 0; i < n_q; i++) {
    Q(i + n_q, i + n_q) = 10;
  }
  trajopt->AddRunningCost(x.transpose()*Q * x);

  // initial guess
  if (!init_file.empty()) {
    MatrixXd z0 = readCSV(data_directory + init_file);
    trajopt->SetInitialGuessForAllVariables(z0);
  } else {
    // Do inverse kinematics to get q initial guess
    vector<VectorXd> q_seed = GetInitGuessForQ(N, stride_length, plant);
    // Do finite differencing to get v initial guess
    vector<VectorXd> v_seed = GetInitGuessForV(q_seed, duration / (N - 1), plant);

    for (int i = 0; i < N; i++) {
      auto xi = trajopt->state(i);
      VectorXd xi_seed(n_q + n_v);
      xi_seed << q_seed.at(i), v_seed.at(i);
      trajopt->SetInitialGuess(xi, xi_seed);
    }
  }
  // Careful: MUST set the initial guess for quaternion, since 0-norm quaterion
  // produces NAN value in some calculation.
  for (int i = 0; i < N; i++) {
    auto xi = trajopt->state(i);
    if (trajopt->GetInitialGuess(xi.head(4)).norm() == 0) {
      trajopt->SetInitialGuess(xi(0), 1);
    }
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

  // Check which solver we are using
  cout << "Solver: " << result.get_solver_id().name() << endl;

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

  return;
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

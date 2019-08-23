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
#include "drake/solvers/choose_best_solver.h"

#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"

#include "drake/multibody/rigid_body_tree_construction.h"
#include "drake/multibody/rigid_body_tree.h"
#include "drake/multibody/rigid_body_plant/drake_visualizer.h"
#include "drake/systems/primitives/trajectory_source.h"

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

#include "examples/Cassie/cassie_utils.h"

#include "attic/multibody/multibody_solvers.h"

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
using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
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

using dairlib::systems::trajectory_optimization::HybridDircon;
using dairlib::systems::trajectory_optimization::DirconDynamicConstraint;
using dairlib::systems::trajectory_optimization::DirconKinematicConstraint;
using dairlib::systems::trajectory_optimization::DirconOptions;
using dairlib::systems::trajectory_optimization::DirconKinConstraintType;
using dairlib::systems::trajectory_optimization::DirconAbstractConstraint;

using dairlib::systems::SubvectorPassThrough;

using dairlib::goldilocks_models::readCSV;
using dairlib::goldilocks_models::writeCSV;

using dairlib::multibody::GetBodyIndexFromName;
using dairlib::multibody::ContactInfo;
using dairlib::multibody::FixedPointSolver;

DEFINE_string(init_file, "", "the file name of initial guess");
DEFINE_int32(max_iter, 500, "Iteration limit");
DEFINE_double(duration_ss, 0.1, "Duration of the single support phase (s)");
DEFINE_double(stride_length, 0.2, "Duration of the walking gait (s)");

namespace dairlib {

/// Currently, MBP doesn't support close-loop linkage, so we add distance
/// constraints in trajectory optimization (dircon).
/// This file runs trajectory optimization for fixed-spring cassie

// Use fixed-point solver to get configuration guess (for standing in place)
void GetInitFixedPointGuess(const Vector3d& pelvis_position,
                            const RigidBodyTree<double>& tree,
                            VectorXd* q_init,
                            VectorXd* u_init,
                            VectorXd* lambda_init) {
  int n_q = tree.get_num_positions();
  int n_v = tree.get_num_velocities();
  int n_u = tree.get_num_actuators();

  int toe_left_idx = GetBodyIndexFromName(tree, "toe_left");
  int toe_right_idx = GetBodyIndexFromName(tree, "toe_right");
  Vector3d pt_front_contact(-0.0457, 0.112, 0);
  Vector3d pt_rear_contact(0.088, 0, 0);
  MatrixXd xa(3,4);
  xa.col(0) = pt_front_contact;
  xa.col(1) = pt_rear_contact;
  xa.col(2) = pt_front_contact;
  xa.col(3) = pt_rear_contact;
  std::vector<int> idxa;
  idxa.push_back(toe_left_idx);
  idxa.push_back(toe_left_idx);
  idxa.push_back(toe_right_idx);
  idxa.push_back(toe_right_idx);
  ContactInfo contact_info(xa, idxa);

  VectorXd q_desired = VectorXd::Zero(n_q);
  q_desired << -0.0872062,
      1.56883E-13,
      1,
      1,
      1.9884E-13,
      1.30167E-14,
      -2.10728E-14,
      -0.0104994,
      0.0104994,
      -3.42713E-07,
      -3.30511E-07,
      0.509601,
      0.509602,
      -1.2219,
      -1.22191,
      1.44602,
      1.44602,
      -1.6072,
      -1.6072;

  std::map<int, double> fixed_joints;
  // for (int i = 0; i < 3; i++) {
  //   fixed_joints[i] = pelvis_position[i];
  // }
  fixed_joints[3] = 1;
  fixed_joints[4] = 0;
  fixed_joints[5] = 0;
  fixed_joints[6] = 0;

  FixedPointSolver fp_solver(tree, contact_info, q_desired, VectorXd::Zero(n_u),
                   MatrixXd::Zero(n_q, n_q), MatrixXd::Identity(n_u, n_u));
  // fp_solver.AddUnitQuaternionConstraint(3, 4, 5, 6);
  fp_solver.AddFrictionConeConstraint(0.8);
  fp_solver.AddJointLimitConstraint(0.1);
  fp_solver.AddFixedJointsConstraint(fixed_joints);
  fp_solver.SetInitialGuessQ(q_desired);
  fp_solver.AddSpreadNormalForcesCost();

  cout << "Solving for fixed point...\n";
  const auto result = fp_solver.Solve();
  SolutionResult solution_result = result.get_solution_result();
  cout << to_string(solution_result) << endl;

  VectorXd q_sol = fp_solver.GetSolutionQ();
  VectorXd u_sol = fp_solver.GetSolutionU();
  VectorXd lambda_sol = fp_solver.GetSolutionLambda();

  VectorXd q_sol_reorder(n_q);
  q_sol_reorder << q_sol.segment(3, 4),
                   q_sol.segment(0, 3),
                   q_sol.tail(12);
  // Careful that the contact ordering should be consistent with those you set
  // up in DIRCON
  VectorXd lambda_sol_reorder(lambda_sol.size());
  VectorXd lambda_sol_contact = lambda_sol.tail(3 * idxa.size());
  for (int i = 0; i < idxa.size(); i++) {
    // We need to reorder cause contact toolkit's lambda ordering is different
    VectorXd lambda_dummy = lambda_sol_contact.segment(3*i, 3);
    lambda_sol_contact(0 + 3*i) = lambda_dummy(1);
    lambda_sol_contact(1 + 3*i) = -lambda_dummy(2);
    lambda_sol_contact(2 + 3*i) = lambda_dummy(0);
  }
  lambda_sol_reorder << lambda_sol_contact,
                        lambda_sol.head(tree.getNumPositionConstraints());

  *q_init = q_sol_reorder;
  *u_init = u_sol;
  *lambda_init = lambda_sol_reorder;

  // Build temporary diagram for visualization
  VectorXd x(n_q + n_v);
  x << q_sol, VectorXd::Zero(n_v);
  drake::lcm::DrakeLcm lcm;
  drake::systems::DiagramBuilder<double> builder;
  const PiecewisePolynomial<double> pp_xtraj = PiecewisePolynomial<double>(x);
  auto state_source = builder.AddSystem<drake::systems::TrajectorySource>
                      (pp_xtraj);
  auto publisher = builder.AddSystem<drake::systems::DrakeVisualizer>(tree,
                   &lcm);
  publisher->set_publish_period(1.0 / 60.0);
  builder.Connect(state_source->get_output_port(),
                  publisher->get_input_port(0));

  auto diagram = builder.Build();
  drake::systems::Simulator<double> simulator(*diagram);
  simulator.set_target_realtime_rate(1);
  simulator.Initialize();
  simulator.StepTo(0.5);
}

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
    // cout << "  Cost:" << result.get_optimal_cost() << std::endl;
    const auto q_sol = result.GetSolution(ik.q());
    cout << "  q_sol = " << q_sol.transpose() << endl;
    VectorXd q_sol_normd(n_q);
    q_sol_normd << q_sol.head(4).normalized(), q_sol.tail(n_q - 4);
    // cout << "  q_sol_normd = " << q_sol_normd << endl;
    q_ik_guess = q_sol_normd;
    q_init_guess.push_back(q_sol_normd);

    /*// Build temporary diagram for visualization
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
    simulator.StepTo(1.0 / N);*/
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
    } else if ( i == q_seed.size() - 1) {
      qdot_seed.push_back((q_seed[i] - q_seed[i - 1]) / dt);
    } else {
      VectorXd v_plus = (q_seed[i + 1] - q_seed[i]) / dt;
      VectorXd v_minus = (q_seed[i] - q_seed[i - 1]) / dt;
      qdot_seed.push_back((v_plus + v_minus) / 2);
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

vector<VectorXd> GetApproxVdot(const vector<VectorXd>& v_seed, double dt,
                               const MultibodyPlant<double>& plant) {
  vector<VectorXd> vdot_approx;
  for (unsigned int i = 0; i < v_seed.size(); i++) {
    if (i == 0) {
      vdot_approx.push_back((v_seed[i + 1] - v_seed[i]) / dt);
    } else {
      vdot_approx.push_back((v_seed[i] - v_seed[i - 1]) / dt);
    }
  }
  return vdot_approx;
}

void GetInitGuessForUAndLambda(const MultibodyPlant<double>& plant,
                               DirconKinematicDataSet<double>& left_dataset,
                               const vector<VectorXd>& q_seed,
                               const vector<VectorXd>& v_seed,
                               const vector<VectorXd>& vdot_approx,
                               vector<VectorXd>* u_seed,
                               vector<VectorXd>* lambda_seed) {
  cout << "\nGetInitGuessForUAndLambda...\n";
  for (unsigned int i = 0; i < q_seed.size(); i++) {
    cout << "i = " << i << endl;
    int n_q = plant.num_positions();
    int n_v = plant.num_velocities();
    int n_u = plant.num_actuators();

    auto context = plant.CreateDefaultContext();
    VectorXd x_seed(n_q + n_v);
    x_seed << q_seed[i], v_seed[i];
    plant.SetPositionsAndVelocities(context.get(), x_seed);

    MatrixXd M = MatrixXd::Zero(n_v, n_v);
    plant.CalcMassMatrixViaInverseDynamics(*context, &M);
    VectorXd f_cg = VectorXd::Zero(n_v);
    plant.CalcBiasTerm(*context, &f_cg);
    f_cg -= plant.CalcGravityGeneralizedForces(*context);
    MatrixXd B = plant.MakeActuationMatrix();

    MatrixXd J = MatrixXd::Zero(left_dataset.countConstraints(), n_v);
    VectorXd JdotV = VectorXd::Zero(left_dataset.countConstraints());
    int index = 0;
    for (int j = 0; j < left_dataset.getNumConstraintObjects(); j++) {
      DirconKinematicData<double>* cj = left_dataset.getConstraint(j);
      cj->updateConstraint(*context);
      int n = cj->getLength();
      J.block(index, 0, n, n_v) = cj->getJ();
      JdotV.segment(index, n) = cj->getJdotv();
      index += n;
    }

    MathematicalProgram quadprog;
    auto u = quadprog.NewContinuousVariables(n_u, "u");
    auto lambda = quadprog.NewContinuousVariables(
                    left_dataset.countConstraints(), "lambda");
    MatrixXd A(n_v, n_u + left_dataset.countConstraints());
    A << B, J.transpose();
    VectorXd b = M * vdot_approx[i] + f_cg;
    quadprog.AddL2NormCost(A, b, {u, lambda});
    const auto result = Solve(quadprog);
    auto solution_result = result.get_solution_result();
    cout << solution_result << endl;
    cout << "  Cost:" << result.get_optimal_cost() << endl;
    VectorXd u_sol = result.GetSolution(u);
    VectorXd lambda_sol = result.GetSolution(lambda);
    cout << "  u_sol = " << u_sol.transpose() << endl;
    cout << "  lambda_sol = " << lambda_sol.transpose() << endl;
  }
}


class QuaternionNormConstraint : public DirconAbstractConstraint<double> {
 public:
  QuaternionNormConstraint() : DirconAbstractConstraint<double>(1, 4,
                                   VectorXd::Zero(1), VectorXd::Zero(1)) {
  }
  ~QuaternionNormConstraint() override = default;

  void EvaluateConstraint(const Eigen::Ref<const drake::VectorX<double>>& x,
                          drake::VectorX<double>* y) const override {
    VectorX<double> output(1);
    output << x.norm() - 1;
    *y = output;
  };
};


void DoMain(double stride_length, double duration_ss, int iter,
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
  // parameters
  bool is_quaterion = false;
  bool standing = false;
  int walking_mode = 1; // 0: instant change of support
                        // 1: single double single
                        // 2: heel to toe
  if (standing) walking_mode = -1;
  bool two_contact_pt_for_walking = true;
  if (walking_mode == 2) two_contact_pt_for_walking = false;
  bool set_second_contact_manually = false;
  bool set_both_contact_pos_manually = false;

  // Scaling paramters
  double trans_pos_scale = 1;
  double rot_pos_scale = 1;
  double omega_scale = 10;
  double input_scale = 100;
  double force_scale = 400;
  double time_scale = 0.03;
  vector<double> var_scale = {omega_scale, input_scale, force_scale, time_scale};

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
  std::vector<bool> row_idx_set_to_0(3, false);
  row_idx_set_to_0[0] = true;
  auto left_toe_rear_indpt_constraint = DirconPositionData<double>(plant, toe_left,
      pt_rear_contact, isXZ, Eigen::Vector2d::Zero(), false, row_idx_set_to_0);
  auto right_toe_rear_indpt_constraint = DirconPositionData<double>(plant, toe_right,
      pt_rear_contact, isXZ, Eigen::Vector2d::Zero(), false, row_idx_set_to_0);
  left_toe_rear_indpt_constraint.addFixedNormalFrictionConstraints(normal, mu);
  right_toe_rear_indpt_constraint.addFixedNormalFrictionConstraints(normal, mu);

  // Testing
  bool isYZ = true; if (isYZ) isXZ = true;
  auto left_toe_rear_2d_constraint = DirconPositionData<double>(plant, toe_left,
        pt_rear_contact, isXZ, Eigen::Vector2d::Zero(), isYZ);
  auto right_toe_rear_2d_constraint = DirconPositionData<double>(plant, toe_right,
        pt_rear_contact, isXZ, Eigen::Vector2d::Zero(), isYZ);
  left_toe_rear_2d_constraint.addFixedNormalFrictionConstraints(normal, mu);
  right_toe_rear_2d_constraint.addFixedNormalFrictionConstraints(normal, mu);

  // Testing (mid contact point)
  isXZ = false;
  Vector3d pt_mid_contact = pt_front_contact + pt_rear_contact;
  auto left_toe_mid_constraint = DirconPositionData<double>(plant, toe_left,
                                 pt_mid_contact, isXZ);
  auto right_toe_mid_constraint = DirconPositionData<double>(plant, toe_right,
                                  pt_mid_contact, isXZ);
  left_toe_mid_constraint.addFixedNormalFrictionConstraints(normal, mu);
  right_toe_mid_constraint.addFixedNormalFrictionConstraints(normal, mu);

  // Compose different types of stance (we call front contact toe and rear
  // contact heel here)
  // left stance (left heel and toe)
  vector<DirconKinematicData<double>*> left_stance_ht_constraint;
  left_stance_ht_constraint.push_back(&left_toe_front_constraint);
  left_stance_ht_constraint.push_back(&left_toe_rear_constraint);
  left_stance_ht_constraint.push_back(&distance_constraint_left);
  left_stance_ht_constraint.push_back(&distance_constraint_right);
  auto left_ht_dataset = DirconKinematicDataSet<double>(plant,
                      &left_stance_ht_constraint);
  auto left_ht_options = DirconOptions(left_ht_dataset.countConstraints());
  left_ht_options.setConstraintRelative(0, true);
  left_ht_options.setConstraintRelative(1, true);
  left_ht_options.setConstraintRelative(3, true);
  left_ht_options.setConstraintRelative(4, true);

  // left stance (left toe)
  vector<DirconKinematicData<double>*> left_stance_t_constraint;
  left_stance_t_constraint.push_back(&left_toe_front_constraint);
  left_stance_t_constraint.push_back(&distance_constraint_left);
  left_stance_t_constraint.push_back(&distance_constraint_right);
  auto left_t_dataset = DirconKinematicDataSet<double>(plant,
                      &left_stance_t_constraint);
  auto left_t_options = DirconOptions(left_t_dataset.countConstraints());
  left_t_options.setConstraintRelative(0, true);
  left_t_options.setConstraintRelative(1, true);

  // Double stance (left toe, right heel)
  vector<DirconKinematicData<double>*> double_stance_th_constraint;
  double_stance_th_constraint.push_back(&left_toe_front_constraint);
  double_stance_th_constraint.push_back(&right_toe_rear_constraint);
  double_stance_th_constraint.push_back(&distance_constraint_left);
  double_stance_th_constraint.push_back(&distance_constraint_right);
  auto double_th_dataset = DirconKinematicDataSet<double>(plant,
                        &double_stance_th_constraint);
  auto double_th_options = DirconOptions(double_th_dataset.countConstraints());
  double_th_options.setConstraintRelative(0, true);
  double_th_options.setConstraintRelative(1, true);
  double_th_options.setConstraintRelative(3, true);
  double_th_options.setConstraintRelative(4, true);

  // Double stance (left toe, right heel and toe)
  vector<DirconKinematicData<double>*> double_stance_tht_constraint;
  double_stance_tht_constraint.push_back(&left_toe_front_constraint);
  double_stance_tht_constraint.push_back(&right_toe_front_constraint);
  double_stance_tht_constraint.push_back(&right_toe_rear_constraint);
  double_stance_tht_constraint.push_back(&distance_constraint_left);
  double_stance_tht_constraint.push_back(&distance_constraint_right);
  auto double_tht_dataset = DirconKinematicDataSet<double>(plant,
                        &double_stance_tht_constraint);
  auto double_tht_options = DirconOptions(double_tht_dataset.countConstraints());
  double_tht_options.setConstraintRelative(0, true);
  double_tht_options.setConstraintRelative(1, true);
  double_tht_options.setConstraintRelative(3, true);
  double_tht_options.setConstraintRelative(4, true);
  double_tht_options.setConstraintRelative(6, true);
  double_tht_options.setConstraintRelative(7, true);

  // right stance (right heel and toe)
  vector<DirconKinematicData<double>*> right_stance_ht_constraint;
  right_stance_ht_constraint.push_back(&right_toe_front_constraint);
  right_stance_ht_constraint.push_back(&right_toe_rear_constraint);
  right_stance_ht_constraint.push_back(&distance_constraint_left);
  right_stance_ht_constraint.push_back(&distance_constraint_right);
  auto right_ht_dataset = DirconKinematicDataSet<double>(plant,
                       &right_stance_ht_constraint);
  auto right_ht_options = DirconOptions(right_ht_dataset.countConstraints());
  right_ht_options.setConstraintRelative(0, true);
  right_ht_options.setConstraintRelative(1, true);
  right_ht_options.setConstraintRelative(3, true);
  right_ht_options.setConstraintRelative(4, true);

  // Double stance all four contacts
  vector<DirconKinematicData<double>*> double_stance_all_constraint;
  double_stance_all_constraint.push_back(&left_toe_front_constraint);
  double_stance_all_constraint.push_back(&left_toe_rear_indpt_constraint);
  double_stance_all_constraint.push_back(&right_toe_front_constraint);
  double_stance_all_constraint.push_back(&right_toe_rear_indpt_constraint);
  double_stance_all_constraint.push_back(&distance_constraint_left);
  double_stance_all_constraint.push_back(&distance_constraint_right);
  auto double_all_dataset = DirconKinematicDataSet<double>(plant,
                        &double_stance_all_constraint);
  auto double_all_options = DirconOptions(double_all_dataset.countConstraints());
  double dist = (pt_front_contact - pt_rear_contact).norm();
  if (set_both_contact_pos_manually) {
    double_all_options.setConstraintRelative(0, false);
    double_all_options.setConstraintRelative(1, false);
    double_all_options.setPhiValue(0, 0);
    double_all_options.setPhiValue(1, 0.12);
    double_all_options.setConstraintRelative(3, false);
    double_all_options.setConstraintRelative(4, false);
    double_all_options.setPhiValue(3, -dist);
    double_all_options.setPhiValue(4, 0.12);
    double_all_options.setConstraintRelative(6, false);
    double_all_options.setConstraintRelative(7, false);
    double_all_options.setPhiValue(6, 0);
    double_all_options.setPhiValue(7, -0.12);
    double_all_options.setConstraintRelative(9, false);
    double_all_options.setConstraintRelative(10, false);
    double_all_options.setPhiValue(9, -dist);
    double_all_options.setPhiValue(10, -0.12);
  } else if (set_second_contact_manually) {
    double_all_options.setConstraintRelative(0, false);
    double_all_options.setConstraintRelative(1, false);
    double_all_options.setPhiValue(0, 0);
    double_all_options.setPhiValue(1, 0.12);
    double_all_options.setConstraintRelative(3, true);
    double_all_options.setConstraintRelative(4, true);
    double_all_options.setConstraintRelative(6, false);
    double_all_options.setConstraintRelative(7, false);
    double_all_options.setPhiValue(6, 0);
    double_all_options.setPhiValue(7, -0.12);
    double_all_options.setConstraintRelative(9, true);
    double_all_options.setConstraintRelative(10, true);
  } else {
    double_all_options.setConstraintRelative(0, true);
    double_all_options.setConstraintRelative(1, true);
    double_all_options.setConstraintRelative(3, true);
    double_all_options.setConstraintRelative(4, true);
    double_all_options.setConstraintRelative(6, true);
    double_all_options.setConstraintRelative(7, true);
    double_all_options.setConstraintRelative(9, true);
    double_all_options.setConstraintRelative(10, true);
  }

  // Testing
  vector<DirconKinematicData<double>*> double_stance_all_2d_constraint;
  double_stance_all_2d_constraint.push_back(&left_toe_front_constraint);
  double_stance_all_2d_constraint.push_back(&left_toe_rear_2d_constraint);
  double_stance_all_2d_constraint.push_back(&right_toe_front_constraint);
  double_stance_all_2d_constraint.push_back(&right_toe_rear_2d_constraint);
  double_stance_all_2d_constraint.push_back(&distance_constraint_left);
  double_stance_all_2d_constraint.push_back(&distance_constraint_right);
  auto double_all_2d_dataset = DirconKinematicDataSet<double>(plant,
                        &double_stance_all_2d_constraint);
  auto double_all_2d_options = DirconOptions(double_all_2d_dataset.countConstraints());
  if (set_both_contact_pos_manually) {
    double_all_2d_options.setConstraintRelative(0, false);
    double_all_2d_options.setConstraintRelative(1, false);
    double_all_2d_options.setPhiValue(0, 0);
    double_all_2d_options.setPhiValue(1, 0.12);
    double_all_2d_options.setConstraintRelative(3, false);
    double_all_2d_options.setPhiValue(3, 0.12);
    double_all_2d_options.setConstraintRelative(5, false);
    double_all_2d_options.setConstraintRelative(6, false);
    double_all_2d_options.setPhiValue(5, 0);
    double_all_2d_options.setPhiValue(6, -0.12);
    double_all_2d_options.setConstraintRelative(8, false);
    double_all_2d_options.setPhiValue(8, -0.12);
  } else if (set_second_contact_manually) {
    double_all_2d_options.setConstraintRelative(0, false);
    double_all_2d_options.setConstraintRelative(1, false);
    double_all_2d_options.setPhiValue(0, 0);
    double_all_2d_options.setPhiValue(1, 0.12);
    double_all_2d_options.setConstraintRelative(3, true);
    double_all_2d_options.setConstraintRelative(5, false);
    double_all_2d_options.setConstraintRelative(6, false);
    double_all_2d_options.setPhiValue(5, 0);
    double_all_2d_options.setPhiValue(6, -0.12);
    double_all_2d_options.setConstraintRelative(8, true);
  } else {
    double_all_2d_options.setConstraintRelative(0, true);
    double_all_2d_options.setConstraintRelative(1, true);
    double_all_2d_options.setConstraintRelative(3, true);
    double_all_2d_options.setConstraintRelative(5, true);
    double_all_2d_options.setConstraintRelative(6, true);
    double_all_2d_options.setConstraintRelative(8, true);
  }

  // Testing - left stance (one contact point)
  vector<DirconKinematicData<double>*> left_stance_mid_constraint;
  left_stance_mid_constraint.push_back(&left_toe_mid_constraint);
  left_stance_mid_constraint.push_back(&distance_constraint_left);
  left_stance_mid_constraint.push_back(&distance_constraint_right);
  auto left_mid_dataset = DirconKinematicDataSet<double>(plant,
                      &left_stance_mid_constraint);
  auto left_mid_options = DirconOptions(left_mid_dataset.countConstraints());
  left_mid_options.setConstraintRelative(0, true);
  left_mid_options.setConstraintRelative(1, true);

  // Testing - right stance (one contact point)
  vector<DirconKinematicData<double>*> right_stance_mid_constraint;
  right_stance_mid_constraint.push_back(&right_toe_mid_constraint);
  right_stance_mid_constraint.push_back(&distance_constraint_left);
  right_stance_mid_constraint.push_back(&distance_constraint_right);
  auto right_mid_dataset = DirconKinematicDataSet<double>(plant,
                       &right_stance_mid_constraint);
  auto right_mid_options = DirconOptions(right_mid_dataset.countConstraints());
  right_mid_options.setConstraintRelative(0, true);
  right_mid_options.setConstraintRelative(1, true);

  // Testing - double stance (one contact point per leg)
  vector<DirconKinematicData<double>*> double_stance_mid_constraint;
  double_stance_mid_constraint.push_back(&left_toe_mid_constraint);
  double_stance_mid_constraint.push_back(&right_toe_mid_constraint);
  double_stance_mid_constraint.push_back(&distance_constraint_left);
  double_stance_mid_constraint.push_back(&distance_constraint_right);
  auto double_mid_dataset = DirconKinematicDataSet<double>(plant,
                        &double_stance_mid_constraint);
  auto double_mid_options = DirconOptions(double_mid_dataset.countConstraints());
  double_mid_options.setConstraintRelative(0, true);
  double_mid_options.setConstraintRelative(1, true);
  double_mid_options.setConstraintRelative(3, true);
  double_mid_options.setConstraintRelative(4, true);



  // Stated in the MultipleShooting class:
  // This class assumes that there are a fixed number (N) time steps/samples
  // and that the trajectory is discretized into timesteps h (N-1 of these),
  // state x (N of these), and control input u (N of these).
  vector<int> num_time_samples;
  vector<double> min_dt;
  vector<double> max_dt;
  vector<DirconKinematicDataSet<double>*> dataset_list;
  vector<DirconOptions> options_list;
  if (walking_mode == 2) {
    num_time_samples.push_back(10);
  } else {
    num_time_samples.push_back(int(40.0 * duration_ss));  // 40 nodes per second
  }
  // Be careful that the nodes per second cannot be too high be cause you have
  // min_dt bound.
  min_dt.push_back(.01);
  max_dt.push_back(.3);
  if (standing) {  // standing
    dataset_list.push_back(&double_all_dataset);
    options_list.push_back(double_all_options);
    // dataset_list.push_back(&double_all_2d_dataset);
    // options_list.push_back(double_all_2d_options);
  } else {  // walking
    if (walking_mode == 0) {
      dataset_list.push_back(two_contact_pt_for_walking? &left_ht_dataset : &left_mid_dataset);
      options_list.push_back(two_contact_pt_for_walking? left_ht_options : left_mid_options);

      // second phase
      num_time_samples.push_back(1);
      min_dt.push_back(.01);
      max_dt.push_back(.3);
      dataset_list.push_back(two_contact_pt_for_walking? &right_ht_dataset : &right_mid_dataset);
      options_list.push_back(two_contact_pt_for_walking? right_ht_options : right_mid_options);
    } else if (walking_mode == 1) {  // walking with double support transition
      dataset_list.push_back(two_contact_pt_for_walking? &left_ht_dataset : &left_mid_dataset);
      options_list.push_back(two_contact_pt_for_walking? left_ht_options : left_mid_options);

      // second phase
      num_time_samples.push_back(int(10.0 * duration_ss));
      min_dt.push_back(.01);
      max_dt.push_back(.3);
      dataset_list.push_back(two_contact_pt_for_walking? &double_all_dataset : &double_mid_dataset);
      options_list.push_back(two_contact_pt_for_walking? double_all_options : double_mid_options);

      // third phase
      num_time_samples.push_back(1);
      min_dt.push_back(.01);
      max_dt.push_back(.3);
      dataset_list.push_back(two_contact_pt_for_walking? &right_ht_dataset : &right_mid_dataset);
      options_list.push_back(two_contact_pt_for_walking? right_ht_options : right_mid_options);
    } else if (walking_mode == 2) {  // walking with heel to toe transition
      dataset_list.push_back(&left_ht_dataset);
      options_list.push_back(left_ht_options);

      // second phase
      num_time_samples.push_back(6);
      min_dt.push_back(.01);
      max_dt.push_back(.3);
      dataset_list.push_back(&left_t_dataset);
      options_list.push_back(left_t_options);

      // third phase
      num_time_samples.push_back(3);
      min_dt.push_back(.01);
      max_dt.push_back(.3);
      dataset_list.push_back(&double_th_dataset);
      options_list.push_back(double_th_options);

      // fourth phase
      num_time_samples.push_back(2);
      min_dt.push_back(.01);
      max_dt.push_back(.3);
      dataset_list.push_back(&double_tht_dataset);
      options_list.push_back(double_tht_options);

      // fifth phase
      num_time_samples.push_back(1);
      min_dt.push_back(.01);
      max_dt.push_back(.3);
      dataset_list.push_back(&right_ht_dataset);
      options_list.push_back(right_ht_options);
    }
  }


  cout << "options_list.size() = " << options_list.size() << endl;
  for (uint i = 0; i < options_list.size(); i ++) {
    cout << "mode # " << i << endl;
    for (auto member : options_list[i].getConstraintsRelative()) {
      cout << member << ", ";
    }
    cout << endl;
  }

  auto trajopt = std::make_shared<HybridDircon<double>>(plant,
                 num_time_samples, min_dt, max_dt, dataset_list, options_list,
                 is_quaterion,
                 omega_scale, input_scale, force_scale, var_scale);

  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Print file", "snopt.out");
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Major iterations limit", iter);
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Verify level", 0);  // 0
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(), "Scale option",
      2);  // 0 // snopt doc said try 2 if seeing snopta exit 40

  int N = 0;
  for (uint i = 0; i < num_time_samples.size(); i++)
    N += num_time_samples[i];
  N -= num_time_samples.size() - 1;  // because of overlaps between modes
  cout << "N = " << N << endl;
  int N_ss;
  if (walking_mode == 2) {
    N_ss = num_time_samples[0] + num_time_samples[1] - 1;
  } else {
    N_ss = num_time_samples[0];
  }


  // Get the decision varaibles that will be used
  auto u = trajopt->input();
  auto x = trajopt->state();
  auto u0 = trajopt->input(0);
  auto uf = trajopt->input(N - 1);
  auto x0 = trajopt->initial_state();
  auto xf = trajopt->state_vars_by_mode(num_time_samples.size() - 1,
                                        num_time_samples[num_time_samples.size() - 1] - 1);
  // Fix the time duration_ss
  cout << "duration_ss = " << duration_ss << endl;
  if (standing) {
    trajopt->AddDurationBounds(duration_ss, duration_ss);
  }

  // quaterion norm constraint
  if (is_quaterion) {
    auto quat_norm_constraint = std::make_shared<QuaternionNormConstraint>();
    for (int i = 0; i < N; i++) {
      auto xi = trajopt->state(i);
      trajopt->AddConstraint(quat_norm_constraint, xi.head(4));
    }
  }

  // Initial quaterion constraint
  if (standing) {
      trajopt->AddLinearConstraint(x0(positions_map.at("position[0]")) == 1);
      trajopt->AddLinearConstraint(x0(positions_map.at("position[1]")) == 0);
      trajopt->AddLinearConstraint(x0(positions_map.at("position[2]")) == 0);
      trajopt->AddLinearConstraint(x0(positions_map.at("position[3]")) == 0);
      // (testing) Final quaternion
      // trajopt->AddLinearConstraint(xf(positions_map.at("position[0]")) >= 0.1);
      // trajopt->AddLinearConstraint(xf(positions_map.at("position[1]")) == 0);
      // trajopt->AddLinearConstraint(xf(positions_map.at("position[2]")) == 0);
      // trajopt->AddLinearConstraint(xf(positions_map.at("position[3]")) == 0);
      // trajopt->AddConstraintToAllKnotPoints(x(positions_map.at("position[0]")) == 1);
      // trajopt->AddConstraintToAllKnotPoints(x(positions_map.at("position[1]")) == 0);
      // trajopt->AddConstraintToAllKnotPoints(x(positions_map.at("position[2]")) == 0);
      // trajopt->AddConstraintToAllKnotPoints(x(positions_map.at("position[3]")) == 0);
  }

  // x-distance constraint constraints
  if (!standing) {
      trajopt->AddLinearConstraint(x0(positions_map.at("position[4]")) == 0);
      trajopt->AddLinearConstraint(xf(positions_map.at("position[4]")) ==
                                   stride_length);
  }

  // testing(initial floating base)
  // trajopt->AddLinearConstraint(x0(positions_map.at("position[6]")) == 1);
  // trajopt->AddLinearConstraint(x0(n_q + velocities_map.at("velocity[5]")) == 0);

  // Testing (standing in place)
  if (standing) {
      // trajopt->AddLinearConstraint(x0(positions_map.at("position[4]")) == -dist/2);
      // trajopt->AddLinearConstraint(x0(positions_map.at("position[5]")) == 0);
      // trajopt->AddLinearConstraint(x0(positions_map.at("position[6]")) == 1);
      // trajopt->AddLinearConstraint(xf(positions_map.at("position[4]")) == -dist/2);
      // trajopt->AddLinearConstraint(xf(positions_map.at("position[5]")) == 0);
      // trajopt->AddLinearConstraint(xf(positions_map.at("position[6]")) == 1);
      // trajopt->AddConstraintToAllKnotPoints(x(positions_map.at("position[4]")) == -dist/2);
      // trajopt->AddConstraintToAllKnotPoints(x(positions_map.at("position[5]")) == 0);
      // trajopt->AddConstraintToAllKnotPoints(x(positions_map.at("position[6]")) == 1);
      trajopt->AddConstraintToAllKnotPoints(x(n_q + velocities_map.at("velocity[3]")) == 0);
      trajopt->AddConstraintToAllKnotPoints(x(n_q + velocities_map.at("velocity[4]")) == 0);
      trajopt->AddConstraintToAllKnotPoints(x(n_q + velocities_map.at("velocity[5]")) == 0);
      trajopt->AddConstraintToAllKnotPoints(x(n_q + velocities_map.at("velocity[0]")) == 0);
      trajopt->AddConstraintToAllKnotPoints(x(n_q + velocities_map.at("velocity[1]")) == 0);
      trajopt->AddConstraintToAllKnotPoints(x(n_q + velocities_map.at("velocity[2]")) == 0);
  }


  // Periodicity constraints
  vector<std::string> left_joint_names {
    "hip_roll_left",
    "hip_yaw_left",
    "hip_pitch_left",
    "knee_left",
    "ankle_joint_left",
    "toe_left"
  };
  vector<std::string> right_joint_names {
    "hip_roll_right",
    "hip_yaw_right",
    "hip_pitch_right",
    "knee_right",
    "ankle_joint_right",
    "toe_right"
  };
  vector<std::string> left_motor_names {
    "hip_pitch_left_motor",
    "hip_roll_left_motor",
    "hip_yaw_left_motor",
    "knee_left_motor",
    "toe_left_motor"
  };
  vector<std::string> right_motor_names {
    "hip_pitch_right_motor",
    "hip_roll_right_motor",
    "hip_yaw_right_motor",
    "knee_right_motor",
    "toe_right_motor"
  };
  // Floating base (mirror in x-z plane)
  if (!standing) {
    // Floating base periodicity
    trajopt->AddLinearConstraint(x0(positions_map.at("position[0]")) ==
                                 xf(positions_map.at("position[0]")));
    trajopt->AddLinearConstraint(x0(positions_map.at("position[1]")) ==
                                 -xf(positions_map.at("position[1]")));
    trajopt->AddLinearConstraint(x0(positions_map.at("position[2]")) ==
                                 xf(positions_map.at("position[2]")));
    trajopt->AddLinearConstraint(x0(positions_map.at("position[3]")) ==
                                 -xf(positions_map.at("position[3]")));
    trajopt->AddLinearConstraint(x0(positions_map.at("position[5]")) ==
                                 -xf(positions_map.at("position[5]")));
    trajopt->AddLinearConstraint(x0(positions_map.at("position[6]")) ==
                                 xf(positions_map.at("position[6]")));
    trajopt->AddLinearConstraint(
      x0(n_q + velocities_map.at("velocity[0]")) ==
      xf(n_q + velocities_map.at("velocity[0]")));
    trajopt->AddLinearConstraint(
      x0(n_q + velocities_map.at("velocity[1]")) ==
      -xf(n_q + velocities_map.at("velocity[1]")));
    trajopt->AddLinearConstraint(
      x0(n_q + velocities_map.at("velocity[2]")) ==
      xf(n_q + velocities_map.at("velocity[2]")));
    trajopt->AddLinearConstraint(
      x0(n_q + velocities_map.at("velocity[3]")) ==
      xf(n_q + velocities_map.at("velocity[3]")));
    trajopt->AddLinearConstraint(
      x0(n_q + velocities_map.at("velocity[4]")) ==
      -xf(n_q + velocities_map.at("velocity[4]")));
    trajopt->AddLinearConstraint(
      x0(n_q + velocities_map.at("velocity[5]")) ==
      xf(n_q + velocities_map.at("velocity[5]")));

    // The legs joint positions and velocities should be mirrored between legs
    for (unsigned int i = 0; i < left_joint_names.size(); i++) {
      trajopt->AddLinearConstraint(x0(positions_map.at(left_joint_names[i])) ==
                                   xf(positions_map.at(right_joint_names[i])));
      trajopt->AddLinearConstraint(x0(positions_map.at(right_joint_names[i])) ==
                                   xf(positions_map.at(left_joint_names[i])));
    }
    for (unsigned int i = 0; i < left_joint_names.size(); i++) {
      trajopt->AddLinearConstraint(
        x0(n_q + velocities_map.at(left_joint_names[i] + "dot")) ==
        xf(n_q + velocities_map.at(right_joint_names[i] + "dot")));
      trajopt->AddLinearConstraint(
        x0(n_q + velocities_map.at(right_joint_names[i] + "dot")) ==
        xf(n_q + velocities_map.at(left_joint_names[i] + "dot")));
    }

    // u periodic constraint
    for (unsigned int i = 0; i < left_motor_names.size(); i++) {
      trajopt->AddLinearConstraint(u0(actuators_map.at(left_motor_names[i])) ==
                                   uf(actuators_map.at(right_motor_names[i])));
      trajopt->AddLinearConstraint(u0(actuators_map.at(right_motor_names[i])) ==
                                   uf(actuators_map.at(left_motor_names[i])));
    }
  }

  // joint limits
  vector<std::string> joint_names {};
  joint_names.insert(joint_names.end(),
                     left_joint_names.begin(), left_joint_names.end() );
  joint_names.insert(joint_names.end(),
                     right_joint_names.begin(), right_joint_names.end() );
  for (const auto & member : joint_names) {
    trajopt->AddConstraintToAllKnotPoints(
      x(positions_map.at(member)) <=
      plant.GetJointByName(member).position_upper_limits()(0));
    trajopt->AddConstraintToAllKnotPoints(
      x(positions_map.at(member)) >=
      plant.GetJointByName(member).position_lower_limits()(0));
  }

  // u limit
  vector<std::string> motor_names {};
  motor_names.insert(motor_names.end(),
                     left_motor_names.begin(), left_motor_names.end() );
  motor_names.insert(motor_names.end(),
                     right_motor_names.begin(), right_motor_names.end() );
  for (const auto & member : motor_names) {
    trajopt->AddConstraintToAllKnotPoints(u(actuators_map.at(member)) <= 300/input_scale);
    trajopt->AddConstraintToAllKnotPoints(u(actuators_map.at(member)) >= -300/input_scale);
  }


  // make sure it's left stance
  // trajopt->AddLinearConstraint(x0(positions_map.at("hip_pitch_left")) <=
  //                              x0(positions_map.at("hip_pitch_right")));









  // // Testing
  // int total_rows = 0;
  // auto constraint_binding_vec = trajopt->GetAllConstraints();
  // for (unsigned int i = 0; i < constraint_binding_vec.size(); i++) {
  //   const auto & binding = constraint_binding_vec[i];
  //   cout << "Constraint row " << total_rows << " to row " <<
  //        total_rows + binding.evaluator()->num_constraints() << ". Vars: " <<
  //        binding.variables().transpose() << endl;
  //   total_rows += binding.evaluator()->num_constraints();
  // }
  // cout << "total_rows = " << total_rows << endl;

  // auto z_all = trajopt->decision_variables();
  // for (int i = 0; i < z_all.size(); i++) {
  //   cout << i << " , " << z_all(i) << endl;
  // }










  // testing (add cost to help convergence postentially?)
  // vector<VectorXd> q_seed = GetInitGuessForQ(N, stride_length, plant);
  // vector<VectorXd> v_seed = GetInitGuessForV(q_seed, duration_ss / (N_ss - 1), plant);
  // MatrixXd S = 10 * MatrixXd::Identity(n_q + n_v, n_q + n_v);
  // for (int i = 0; i < N; i++) {
  //   auto xi = trajopt->state(i);
  //   VectorXd xi_seed(n_q + n_v);
  //   xi_seed << q_seed.at(i), v_seed.at(i);
  //   trajopt->AddQuadraticErrorCost(S, xi_seed, xi);
  // }








  // add cost
  const double R = 10/* * input_scale * input_scale*/;  // Cost on input effort
  MatrixXd Q = 10 * MatrixXd::Identity(n_v, n_v)/* * omega_scale * omega_scale*/;
  trajopt->AddRunningCost(u.transpose()* R * u);
  trajopt->AddRunningCost(x.tail(n_v).transpose()* Q * x.tail(n_v));



  // Testing (add initial guess from xi, ui)
  // MatrixXd x_i_input = readCSV(data_directory + "x_i.csv");
  // MatrixXd u_i_input = readCSV(data_directory + "u_i.csv");
  // for (int i = 0; i < N; i++) {
  //   auto xi = trajopt->state(i);
  //   trajopt->SetInitialGuess(xi, x_i_input.col(i));
  //   auto ui = trajopt->input(i);
  //   trajopt->SetInitialGuess(ui, u_i_input.col(i));
  // }







  // initial guess
  if (!init_file.empty()) {
    MatrixXd z0 = readCSV(data_directory + init_file);
    trajopt->SetInitialGuessForAllVariables(z0);
  } else {
    if (standing) {
      // Use RBT fixed point solver
      RigidBodyTree<double> tree;
      buildCassieTree(tree,
                      "examples/Cassie/urdf/cassie_fixed_springs.urdf",
                      drake::multibody::joints::kQuaternion, false);
      const double terrain_size = 100;
      const double terrain_depth = 0.20;
      drake::multibody::AddFlatTerrainToWorld(&tree,
                                              terrain_size, terrain_depth);

      Vector3d pelvis_position(-dist/2, 0, 1);
      VectorXd q_init;
      VectorXd u_init;
      VectorXd lambda_init;
      GetInitFixedPointGuess(pelvis_position, tree,
                                 &q_init, &u_init, &lambda_init);
      cout << "q_init from fixed-point solver = " << q_init << endl;
      cout << "u_init from fixed-point solver = " << u_init << endl;
      cout << "lambda_init from fixed-point solver = " << lambda_init << endl;

      for (int i = 0; i < N; i++) {
        auto xi = trajopt->state(i);
        VectorXd xi_init(n_q + n_v);
        xi_init << q_init, VectorXd::Zero(n_v);
        trajopt->SetInitialGuess(xi, xi_init);

        auto ui = trajopt->input(i);
        trajopt->SetInitialGuess(ui, u_init / input_scale);
      }
      for (unsigned int mode = 0; mode < num_time_samples.size(); mode++) {
        for (int index = 0; index < num_time_samples[mode]; index++) {
          auto lambdai = trajopt->force(mode, index);
          trajopt->SetInitialGuess(lambdai, lambda_init / force_scale);
        }
      }

    } else {
      // Do inverse kinematics to get q initial guess
      vector<VectorXd> q_seed = GetInitGuessForQ(N_ss, stride_length, plant);
      // Do finite differencing to get v initial guess
      vector<VectorXd> v_seed = GetInitGuessForV(q_seed, duration_ss / (N_ss - 1), plant);
      for (int i = 0; i < N; i++) {
        auto xi = trajopt->state(i);
        VectorXd xi_seed(n_q + n_v);
        if (i < N_ss) {
          xi_seed << q_seed.at(i), v_seed.at(i) / omega_scale;
        } else {
          xi_seed << q_seed.at(N_ss - 1), v_seed.at(N_ss - 1)  / omega_scale;
        }
        trajopt->SetInitialGuess(xi, xi_seed);
      }
      /*
      // Get approximated vdot by finite difference
      vector<VectorXd> vdot_approx = GetApproxVdot(v_seed, duration_ss / (N - 1), plant);
      // Solve QP to get u and lambda
      vector<VectorXd> u_seed(N, VectorXd::Zero(n_u));
      vector<VectorXd> lambda_seed(N, VectorXd::Zero(
                                     left_dataset.countConstraints()));
      GetInitGuessForUAndLambda(plant, left_dataset,
                                q_seed, v_seed, vdot_approx,
                                &u_seed, &lambda_seed);
      for (int i = 0; i < N; i++) {
        auto ui = trajopt->input(i);
        trajopt->SetInitialGuess(ui, u_seed.at(i) / input_scale);

        // trajopt->SetInitialGuess(lambdai, lambda_seed.at(i) / force_scale);
      }
      */
    }
  }
  // Careful: MUST set the initial guess for quaternion, since 0-norm quaterion
  // produces NAN value in some calculation.
  for (int i = 0; i < N; i++) {
    auto xi = trajopt->state(i);
    if ((trajopt->GetInitialGuess(xi.head(4)).norm() == 0) ||
        std::isnan(trajopt->GetInitialGuess(xi.head(4)).norm())) {
      trajopt->SetInitialGuess(xi(0), 1);
      trajopt->SetInitialGuess(xi(1), 0);
      trajopt->SetInitialGuess(xi(2), 0);
      trajopt->SetInitialGuess(xi(3), 0);
    }
  }

  cout << "Choose the best solver: " <<
      drake::solvers::ChooseBestSolver(*trajopt).name() << endl;

  cout << "Solving DIRCON\n";
  cout << '\a';
  auto start = std::chrono::high_resolution_clock::now();
  const auto result = Solve(*trajopt, trajopt->initial_guess());
  SolutionResult solution_result = result.get_solution_result();
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  // trajopt->PrintSolution();
  for (int i = 0; i < 100; i++) {cout << '\a';}
  cout << to_string(solution_result) << endl;
  cout << "Solve time:" << elapsed.count() << std::endl;
  cout << "Cost:" << result.get_optimal_cost() << std::endl;

  // Check which solver we are using
  cout << "Solver: " << result.get_solver_id().name() << endl;

  // store the solution of the decision variable
  VectorXd z = result.GetSolution(trajopt->decision_variables());
  writeCSV(data_directory + output_prefix + string("z.csv"), z);
  // for(int i = 0; i<z.size(); i++){
  //   cout << trajopt->decision_variables()[i] << ", " << z[i] << endl;
  // }
  // cout << endl;

  // store the time, state, and input at knot points
  VectorXd time_at_knots = trajopt->GetSampleTimes(result);
  MatrixXd state_at_knots = trajopt->GetStateSamples(result);
  MatrixXd input_at_knots = trajopt->GetInputSamples(result);
  state_at_knots.col(N-1) = result.GetSolution(xf);
  writeCSV(data_directory + string("t_i.csv"), time_at_knots);
  writeCSV(data_directory + string("x_i.csv"), state_at_knots);
  writeCSV(data_directory + string("u_i.csv"), input_at_knots);
  state_at_knots << state_at_knots.block(0,0,n_q,state_at_knots.cols()),
        state_at_knots.block(n_q,0,n_v,state_at_knots.cols()) * omega_scale;
  cout << "time_at_knots = \n" << time_at_knots << "\n";
  cout << "state_at_knots = \n" << state_at_knots << "\n";
  cout << "state_at_knots.size() = " << state_at_knots.size() << endl;
  cout << "input_at_knots = \n" << input_at_knots * input_scale << "\n";

  // Testing
  cout << "lambda_sol = \n";
  for (unsigned int mode = 0; mode < num_time_samples.size(); mode++) {
    for (int index = 0; index < num_time_samples[mode]; index++) {
      auto lambdai = trajopt->force(mode, index);
      cout << result.GetSolution(lambdai).transpose() * force_scale << endl;
    }
  }

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


int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  double stride_length = FLAGS_stride_length;
  double duration_ss = FLAGS_duration_ss; //0.5
  int iter = FLAGS_max_iter;
  string data_directory = "examples/Cassie/trajopt_data/";
  string init_file = FLAGS_init_file;
  // string init_file = "testing_z.csv";
  string output_prefix = "";

  dairlib::DoMain(stride_length, duration_ss, iter,
                  data_directory, init_file, output_prefix);
}

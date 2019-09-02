#include <gflags/gflags.h>

#include <memory>
#include <chrono>
#include <string>
#include <fstream>

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
using drake::multibody::BodyIndex;
using drake::multibody::ModelInstanceIndex;

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
DEFINE_int32(max_iter, 10000, "Iteration limit");
DEFINE_double(duration, 0.4, "Duration of the single support phase (s)");
DEFINE_double(stride_length, 0.2, "Duration of the walking gait (s)");
DEFINE_double(ground_incline, 0.0, "Duration of the walking gait (s)");
DEFINE_double(omega_scale, 10, "Variable scaling");
DEFINE_double(input_scale, 100, "Variable scaling");
DEFINE_double(force_scale, 1000, "Variable scaling");
DEFINE_double(time_scale, 0.008, "Variable scaling");
DEFINE_double(quaternion_scale, 0.5, "Variable scaling");
DEFINE_double(tol, 1e-4,
              "Tolerance for constraint violation and dual gap");

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
  MatrixXd xa(3, 4);
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
  for (unsigned int i = 0; i < idxa.size(); i++) {
    // We need to reorder cause contact toolkit's lambda ordering is different
    VectorXd lambda_dummy = lambda_sol_contact.segment(3 * i, 3);
    lambda_sol_contact(0 + 3 * i) = lambda_dummy(1);
    lambda_sol_contact(1 + 3 * i) = -lambda_dummy(2);
    lambda_sol_contact(2 + 3 * i) = lambda_dummy(0);
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
    // SolutionResult solution_result = result.get_solution_result();
    // cout << "\n" << to_string(solution_result) << endl;
    // cout << "  Cost:" << result.get_optimal_cost() << std::endl;
    const auto q_sol = result.GetSolution(ik.q());
    // cout << "  q_sol = " << q_sol.transpose() << endl;
    VectorXd q_sol_normd(n_q);
    q_sol_normd << q_sol.head(4).normalized(), q_sol.tail(n_q - 4);
    // cout << "  q_sol_normd = " << q_sol_normd << endl;
    q_ik_guess = q_sol_normd;
    q_init_guess.push_back(q_sol_normd);

    bool visualize_init_traj = false;
    if (visualize_init_traj) {
      // Build temporary diagram for visualization
      drake::systems::DiagramBuilder<double> builder_ik;
      SceneGraph<double>& scene_graph_ik = *builder_ik.AddSystem<SceneGraph>();
      scene_graph_ik.set_name("scene_graph_ik");
      MultibodyPlant<double> plant_ik;
      multibody::addFlatTerrain(&plant_ik, &scene_graph_ik, .8, .8);
      Parser parser(&plant_ik, &scene_graph_ik);
      string full_name =
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
    // cout << i << ":\n";
    // cout << "  qdot = " << qdot_seed[i].transpose() << endl;
    // cout << "  v = " << v.transpose() << endl;
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
  QuaternionNormConstraint(vector<double> var_scale) :
    DirconAbstractConstraint<double>(1, 4,
                                     VectorXd::Zero(1), VectorXd::Zero(1),
                                     "quaternion_norm_constraint"),
    quaternion_scale_(var_scale[4]) {
  }
  ~QuaternionNormConstraint() override = default;

  void EvaluateConstraint(const Eigen::Ref<const drake::VectorX<double>>& x,
                          drake::VectorX<double>* y) const override {
    VectorX<double> output(1);
    output << quaternion_scale_ * x.norm() - 1;
    *y = output;
  };
 private:
  double quaternion_scale_;
};

// class RedundantForceConstraint : public DirconAbstractConstraint<double> {
//  public:
//   RedundantForceConstraint() : DirconAbstractConstraint<double>(1, 2,
//       VectorXd::Zero(1),
//       VectorXd::Constant(1, std::numeric_limits<double>::infinity()),
//       "redundant_force_constraint") {
//   }
//   ~RedundantForceConstraint() override = default;

//   void EvaluateConstraint(const Eigen::Ref<const drake::VectorX<double>>& x,
//                           drake::VectorX<double>* y) const override {
//     *y = x.head(1) * x.tail(1);
//   };
// };

class ComHeightConstraint : public DirconAbstractConstraint<double> {
 public:
  ComHeightConstraint(const MultibodyPlant<double>* plant,
                      vector<double> var_scale) :
    DirconAbstractConstraint<double>(
      1, 2 * (plant->num_positions() + plant->num_velocities()),
      VectorXd::Zero(1), VectorXd::Zero(1),
      "com_height_constraint"),
    plant_(plant),
    n_q_(plant->num_positions()),
    n_v_(plant->num_velocities()),
    quaternion_scale_(var_scale[4]) {
  }
  ~ComHeightConstraint() override = default;

  void EvaluateConstraint(const Eigen::Ref<const drake::VectorX<double>>& x,
                          drake::VectorX<double>* y) const override {
    VectorXd q1 = x.head(n_q_);
    VectorXd q2 = x.segment(n_q_ + n_v_, n_q_);
    q1.head(4) *= quaternion_scale_;
    q2.head(4) *= quaternion_scale_;

    std::unique_ptr<drake::systems::Context<double>> context =
          plant_->CreateDefaultContext();
    plant_->SetPositions(context.get(), q1);
    VectorXd CoM_z1 = plant_->CalcCenterOfMassPosition(*context).tail(1);
    plant_->SetPositions(context.get(), q2);
    VectorXd CoM_z2 = plant_->CalcCenterOfMassPosition(*context).tail(1);

    *y = CoM_z1 - CoM_z2;
  };
 private:
  const MultibodyPlant<double>* plant_;
  int n_q_;
  int n_v_;
  double quaternion_scale_;
};

class ComHeightVelConstraint : public DirconAbstractConstraint<double> {
 public:
  ComHeightVelConstraint(const MultibodyPlant<double>* plant,
                         vector<double> var_scale) :
    DirconAbstractConstraint<double>(
      1, plant->num_positions() + plant->num_velocities(),
      VectorXd::Zero(1), VectorXd::Zero(1),
      "com_height_vel_constraint"),
    plant_(plant),
    n_q_(plant->num_positions()),
    n_v_(plant->num_velocities()),
    omega_scale_(var_scale[0]),
    quaternion_scale_(var_scale[4]) {

    DRAKE_DEMAND(plant->num_bodies() > 1);
    DRAKE_DEMAND(plant->num_model_instances() > 1);

    // Get all body indices
    std::vector<ModelInstanceIndex> model_instances;
    for (ModelInstanceIndex model_instance_index(1);
         model_instance_index < plant->num_model_instances();
         ++model_instance_index)
      model_instances.push_back(model_instance_index);
    for (auto model_instance : model_instances) {
      const std::vector<BodyIndex> body_index_in_instance =
        plant->GetBodyIndices(model_instance);
      for (BodyIndex body_index : body_index_in_instance)
        body_indexes_.push_back(body_index);
    }
    // Get total mass
    std::unique_ptr<drake::systems::Context<double>> context =
          plant->CreateDefaultContext();
    for (BodyIndex body_index : body_indexes_) {
      if (body_index == 0) continue;
      const Body<double>& body = plant_->get_body(body_index);

      // Calculate composite_mass_.
      const double& body_mass = body.get_mass(*context);
      // composite_mass_ = ∑ mᵢ
      composite_mass_ += body_mass;
    }
    if (!(composite_mass_ > 0)) {
      throw std::runtime_error(
        "The total mass must larger than zero.");
    }
  }
  ~ComHeightVelConstraint() override = default;

  void EvaluateConstraint(const Eigen::Ref<const drake::VectorX<double>>& x,
                          drake::VectorX<double>* y) const override {
    VectorXd q = x.head(n_q_);
    q.head(4) *= quaternion_scale_;
    VectorXd v = x.tail(n_v_) * omega_scale_;

    std::unique_ptr<drake::systems::Context<double>> context =
          plant_->CreateDefaultContext();
    plant_->SetPositions(context.get(), q);
    plant_->SetVelocities(context.get(), v);

    const drake::multibody::Frame<double>& world = plant_->world_frame();

    // Get com jacobian
    MatrixXd Jcom = MatrixXd::Zero(3, n_v_);
    for (BodyIndex body_index : body_indexes_) {
      if (body_index == 0) continue;

      const Body<double>& body = plant_->get_body(body_index);
      const Vector3d pi_BoBcm = body.CalcCenterOfMassInBodyFrame(*context);

      // Calculate M * J in world frame.
      const double& body_mass = body.get_mass(*context);
      // Jcom = ∑ mᵢ * Ji
      MatrixXd Jcom_i(3, n_v_);
      plant_->CalcJacobianTranslationalVelocity(
        *context, drake::multibody::JacobianWrtVariable::kV,
        body.body_frame(), pi_BoBcm, world, world, &Jcom_i);
      Jcom += body_mass * Jcom_i;
      // cout << "body_mass = " << body_mass << endl;
      // cout << "Jcom_i = " << Jcom_i << endl;
    }
    Jcom /= composite_mass_;

    *y = Jcom.row(2) * v;
  };
 private:
  const MultibodyPlant<double>* plant_;
  int n_q_;
  int n_v_;
  double omega_scale_;
  double quaternion_scale_;

  std::vector<BodyIndex> body_indexes_;
  double composite_mass_;
};


void DoMain(double stride_length,
            double ground_incline,
            double duration, int iter,
            string data_directory,
            string init_file,
            string output_prefix,
            double omega_scale,
            double input_scale,
            double force_scale,
            double time_scale,
            double quaternion_scale,
            double tol) {
  drake::systems::DiagramBuilder<double> builder;
  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  scene_graph.set_name("scene_graph");

  MultibodyPlant<double> plant;
  // multibody::addFlatTerrain(&plant, &scene_graph, .8, .8);
  Parser parser(&plant, &scene_graph);

  string full_name =
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
  bool is_quaterion = true;
  bool standing = false;
  int walking_mode = 0; // 0: instant change of support
  // 1: single double single
  // 2: heel to toe
  if (standing) walking_mode = -1;
  bool set_second_contact_manually = false;
  bool set_both_contact_pos_manually = false;

  // Scaling paramters
  // double omega_scale = 10;  // 10
  // double input_scale = 100;
  // double force_scale = 1000;  // 400
  // double time_scale = 0.008;  // 0.01
  // double quaternion_scale = 0.5;  // 1
  // double trans_pos_scale = 1;
  // double rot_pos_scale = 1;
  vector<double> var_scale = {omega_scale, input_scale, force_scale, time_scale,
                              quaternion_scale
                             };

  const Body<double>& toe_left = plant.GetBodyByName("toe_left");
  const Body<double>& toe_right = plant.GetBodyByName("toe_right");
  Vector3d pt_front_contact(-0.0457, 0.112, 0);
  Vector3d pt_rear_contact(0.088, 0, 0);
  bool isXZ = false;
  Eigen::Vector2d ground_rp(0, ground_incline);  // gournd incline in roll pitch
  auto left_toe_front_constraint = DirconPositionData<double>(plant, toe_left,
                                   pt_front_contact, isXZ, ground_rp);
  auto left_toe_rear_constraint = DirconPositionData<double>(plant, toe_left,
                                  pt_rear_contact, isXZ, ground_rp);
  auto right_toe_front_constraint = DirconPositionData<double>(plant, toe_right,
                                    pt_front_contact, isXZ, ground_rp);
  auto right_toe_rear_constraint = DirconPositionData<double>(plant, toe_right,
                                   pt_rear_contact, isXZ, ground_rp);
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

  // Relaxed constraint of rear contact point
  std::vector<bool> row_idx_set_to_0(3, false);
  row_idx_set_to_0[0] = true;
  auto left_toe_rear_indpt_constraint = DirconPositionData<double>(plant,
                                        toe_left, pt_rear_contact, isXZ,
                                        ground_rp, false, row_idx_set_to_0);
  auto right_toe_rear_indpt_constraint = DirconPositionData<double>(plant,
                                         toe_right, pt_rear_contact, isXZ,
                                         ground_rp, false, row_idx_set_to_0);
  left_toe_rear_indpt_constraint.addFixedNormalFrictionConstraints(normal, mu);
  right_toe_rear_indpt_constraint.addFixedNormalFrictionConstraints(normal, mu);

  // Compose different types of stance (we call front contact toe and rear
  // contact heel here)
  // left stance (left heel and toe)
  vector<DirconKinematicData<double>*> left_stance_ht_constraint;
  left_stance_ht_constraint.push_back(&left_toe_front_constraint);
  left_stance_ht_constraint.push_back(&left_toe_rear_indpt_constraint);
  left_stance_ht_constraint.push_back(&distance_constraint_left);
  left_stance_ht_constraint.push_back(&distance_constraint_right);
  auto left_ht_dataset = DirconKinematicDataSet<double>(plant,
                         &left_stance_ht_constraint);
  auto left_ht_options = DirconOptions(left_ht_dataset.countConstraints());
  left_ht_options.setConstraintRelative(0, true);
  left_ht_options.setConstraintRelative(1, true);
  left_ht_options.setConstraintRelative(3, true);
  left_ht_options.setConstraintRelative(4, true);
  if (walking_mode == 1) {
    // left_ht_options.setEndType(DirconKinConstraintType::kAccelAndVel);
  }

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
  double_stance_tht_constraint.push_back(&right_toe_rear_indpt_constraint);
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
  right_stance_ht_constraint.push_back(&right_toe_rear_indpt_constraint);
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

  // Stated in the MultipleShooting class:
  vector<int> num_time_samples;
  vector<double> min_dt;
  vector<double> max_dt;
  vector<DirconKinematicDataSet<double>*> dataset_list;
  vector<DirconOptions> options_list;
  // num_time_samples.push_back(int(40.0 * duration));  // 40 nodes per second
  // Be careful that the nodes per second cannot be too high be cause you have
  // min_dt bound.
  min_dt.push_back(.01);
  max_dt.push_back(.3);
  if (standing) {  // standing
    num_time_samples.push_back(int(40.0 * duration));  // 40 nodes per second
    dataset_list.push_back(&double_all_dataset);
    options_list.push_back(double_all_options);
  } else {  // walking
    if (walking_mode == 0) {
      num_time_samples.push_back(16); //16 for 0.4 second
      dataset_list.push_back(&left_ht_dataset);
      options_list.push_back(left_ht_options);

      // second phase
      num_time_samples.push_back(1);
      min_dt.push_back(.01);
      max_dt.push_back(.3);
      dataset_list.push_back(&right_ht_dataset);
      options_list.push_back(right_ht_options);
    } else if (walking_mode == 1) {  // walking with double support transition
      num_time_samples.push_back(16);
      dataset_list.push_back(&left_ht_dataset);
      options_list.push_back(left_ht_options);

      // second phase
      num_time_samples.push_back(4);
      min_dt.push_back(.01);
      max_dt.push_back(.3);
      dataset_list.push_back(&double_all_dataset);
      options_list.push_back(double_all_options);

      // third phase
      num_time_samples.push_back(1);
      min_dt.push_back(.01);
      max_dt.push_back(.3);
      dataset_list.push_back(&right_ht_dataset);
      options_list.push_back(right_ht_options);
    } else if (walking_mode == 2) {  // walking with heel to toe transition
      num_time_samples.push_back(10);
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
                 var_scale);

  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Print file", "snopt.out");
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Major iterations limit", iter);
  // trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
  //                          "Iterations limit", 100000);  // QP subproblems
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Verify level", 0);  // 0
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(), "Scale option",
                           2);  // 0 // snopt doc said try 2 if seeing snopta exit 40
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Major optimality tolerance", tol);  // target nonlinear constraint violation
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Major feasibility tolerance", tol);  // target complementarity gap

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
  // Fix the time duration
  cout << "duration = " << duration << endl;
  // if (standing) {
  //   trajopt->AddDurationBounds(duration / time_scale, duration / time_scale);
  // }

  // Fix time duration
  trajopt->AddDurationBounds(duration / time_scale, duration / time_scale);

  // quaterion norm constraint
  if (is_quaterion) {
    auto quat_norm_constraint = std::make_shared<QuaternionNormConstraint>
                                (var_scale);
    for (int i = 0; i < N; i++) {
      auto xi = trajopt->state(i);
      trajopt->AddConstraint(quat_norm_constraint, xi.head(4));
    }
  }

  // Initial quaterion constraint
  if (standing) {
    // trajopt->AddLinearConstraint(quaternion_scale * x0(positions_map.at("position[0]")) == 1);
    // trajopt->AddLinearConstraint(quaternion_scale * x0(positions_map.at("position[1]")) == 0);
    // trajopt->AddLinearConstraint(quaternion_scale * x0(positions_map.at("position[2]")) == 0);
    // trajopt->AddLinearConstraint(quaternion_scale * x0(positions_map.at("position[3]")) == 0);
    trajopt->AddBoundingBoxConstraint(1 / quaternion_scale, 1 / quaternion_scale,
                                      x0(positions_map.at("position[0]")));
    trajopt->AddBoundingBoxConstraint(0 / quaternion_scale, 0 / quaternion_scale,
                                      x0(positions_map.at("position[1]")));
    trajopt->AddBoundingBoxConstraint(0 / quaternion_scale, 0 / quaternion_scale,
                                      x0(positions_map.at("position[2]")));
    trajopt->AddBoundingBoxConstraint(0 / quaternion_scale, 0 / quaternion_scale,
                                      x0(positions_map.at("position[3]")));
    // (testing) Final quaternion
    // trajopt->AddLinearConstraint(quaternion_scale * xf(positions_map.at("position[0]")) >= 0.1);
    // trajopt->AddLinearConstraint(quaternion_scale * xf(positions_map.at("position[1]")) == 0);
    // trajopt->AddLinearConstraint(quaternion_scale * xf(positions_map.at("position[2]")) == 0);
    // trajopt->AddLinearConstraint(quaternion_scale * xf(positions_map.at("position[3]")) == 0);
    // trajopt->AddConstraintToAllKnotPoints(quaternion_scale * x(positions_map.at("position[0]")) == 1);
    // trajopt->AddConstraintToAllKnotPoints(quaternion_scale * x(positions_map.at("position[1]")) == 0);
    // trajopt->AddConstraintToAllKnotPoints(quaternion_scale * x(positions_map.at("position[2]")) == 0);
    // trajopt->AddConstraintToAllKnotPoints(quaternion_scale * x(positions_map.at("position[3]")) == 0);
  }

  // x-distance constraint constraints
  if (!standing) {
    // trajopt->AddLinearConstraint(x0(positions_map.at("position[4]")) == 0);
    // trajopt->AddLinearConstraint(xf(positions_map.at("position[4]")) ==
    //                              stride_length);
    trajopt->AddBoundingBoxConstraint(0, 0,
                                      x0(positions_map.at("position[4]")));
    trajopt->AddBoundingBoxConstraint(stride_length, stride_length,
                                      xf(positions_map.at("position[4]")));
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
    trajopt->AddConstraintToAllKnotPoints(
      x(n_q + velocities_map.at("velocity[3]")) == 0);
    trajopt->AddConstraintToAllKnotPoints(
      x(n_q + velocities_map.at("velocity[4]")) == 0);
    trajopt->AddConstraintToAllKnotPoints(
      x(n_q + velocities_map.at("velocity[5]")) == 0);
    trajopt->AddConstraintToAllKnotPoints(
      x(n_q + velocities_map.at("velocity[0]")) == 0);
    trajopt->AddConstraintToAllKnotPoints(
      x(n_q + velocities_map.at("velocity[1]")) == 0);
    trajopt->AddConstraintToAllKnotPoints(
      x(n_q + velocities_map.at("velocity[2]")) == 0);
  }


  // Testing - fix com height during walking (only the first mode)
  // The purpose is to get a good seed for RoM traj opt
  auto com_constraint = std::make_shared<ComHeightConstraint>(&plant, var_scale);
  for (int index = 0; index < num_time_samples[0] - 1; index++) {
    auto x0 = trajopt->state(index);
    auto x1 = trajopt->state(index + 1);
    trajopt->AddConstraint(com_constraint, {x0, x1});
  }
  auto com_vel_constraint = std::make_shared<ComHeightVelConstraint>(&plant,
                            var_scale);
  for (int index = 0; index < num_time_samples[0]; index++) {
    auto x = trajopt->state(index);
    trajopt->AddConstraint(com_vel_constraint, x);
  }



  // Periodicity constraints
  // vector<string> left_joint_names {
  //   "hip_roll_left",
  //   "hip_yaw_left",
  //   "hip_pitch_left",
  //   "knee_left",
  //   "ankle_joint_left",
  //   "toe_left"
  // };
  // vector<string> right_joint_names {
  //   "hip_roll_right",
  //   "hip_yaw_right",
  //   "hip_pitch_right",
  //   "knee_right",
  //   "ankle_joint_right",
  //   "toe_right"
  // };
  // vector<string> left_motor_names {
  //   "hip_pitch_left_motor",
  //   "hip_roll_left_motor",
  //   "hip_yaw_left_motor",
  //   "knee_left_motor",
  //   "toe_left_motor"
  // };
  // vector<string> right_motor_names {
  //   "hip_pitch_right_motor",
  //   "hip_roll_right_motor",
  //   "hip_yaw_right_motor",
  //   "knee_right_motor",
  //   "toe_right_motor"
  // };
  vector<std::pair<string, string>> l_r_pairs {
    std::pair<string, string>("_left", "_right"),
    std::pair<string, string>("_right", "_left"),
  };
  vector<string> asy_joint_names {
    "hip_roll",
    "hip_yaw",
  };
  vector<string> sym_joint_names {
    "hip_pitch",
    "knee",
    "ankle_joint",
    "toe"
  };
  vector<string> joint_names {};
  vector<string> motor_names {};
  for (auto l_r_pair : l_r_pairs) {
    for (unsigned int i = 0; i < asy_joint_names.size(); i++) {
      joint_names.push_back(asy_joint_names[i] + l_r_pair.first);
      motor_names.push_back(asy_joint_names[i] + l_r_pair.first + "_motor");
    }
    for (unsigned int i = 0; i < sym_joint_names.size(); i++) {
      joint_names.push_back(sym_joint_names[i] + l_r_pair.first);
      if (sym_joint_names[i].compare("ankle_joint") != 0) {
        motor_names.push_back(sym_joint_names[i] + l_r_pair.first + "_motor");
      }
    }
  }

  //testing - print out names
  // for (auto member : joint_names) {
  //   cout << member << endl;
  // }
  // for (auto member : motor_names) {
  //   cout << member << endl;
  // }

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

    // The legs joint positions/velocities/torque should be mirrored between legs
    // (notice that hip yaw and roll should be asymmetric instead of symmetric.)
    for (auto l_r_pair : l_r_pairs) {
      for (unsigned int i = 0; i < asy_joint_names.size(); i++) {
        // positions
        trajopt->AddLinearConstraint(
          x0(positions_map.at(asy_joint_names[i] + l_r_pair.first)) ==
          -xf(positions_map.at(asy_joint_names[i] + l_r_pair.second)));
        // velocities
        trajopt->AddLinearConstraint(
          x0(n_q + velocities_map.at(asy_joint_names[i] + l_r_pair.first + "dot")) ==
          -xf(n_q + velocities_map.at(asy_joint_names[i] + l_r_pair.second + "dot")));
        // inputs
        trajopt->AddLinearConstraint(
          u0(actuators_map.at(asy_joint_names[i] + l_r_pair.first + "_motor")) ==
          -uf(actuators_map.at(asy_joint_names[i] + l_r_pair.second + "_motor")));
      }
      for (unsigned int i = 0; i < sym_joint_names.size(); i++) {
        // positions
        trajopt->AddLinearConstraint(
          x0(positions_map.at(sym_joint_names[i] + l_r_pair.first)) ==
          xf(positions_map.at(sym_joint_names[i] + l_r_pair.second)));
        // velocities
        trajopt->AddLinearConstraint(
          x0(n_q + velocities_map.at(sym_joint_names[i] + l_r_pair.first + "dot")) ==
          xf(n_q + velocities_map.at(sym_joint_names[i] + l_r_pair.second + "dot")));
        // inputs
        if (sym_joint_names[i].compare("ankle_joint") != 0) {
          trajopt->AddLinearConstraint(
            u0(actuators_map.at(sym_joint_names[i] + l_r_pair.first + "_motor")) ==
            uf(actuators_map.at(sym_joint_names[i] + l_r_pair.second + "_motor")));
        }
      }
    }  // end for (l_r_pairs)
  }  // end if (!standing)

  // joint limits
  for (const auto & member : joint_names) {
    trajopt->AddConstraintToAllKnotPoints(
      x(positions_map.at(member)) <=
      plant.GetJointByName(member).position_upper_limits()(0));
    trajopt->AddConstraintToAllKnotPoints(
      x(positions_map.at(member)) >=
      plant.GetJointByName(member).position_lower_limits()(0));
  }

  // u limit
  // for (const auto & member : motor_names) {
  // trajopt->AddConstraintToAllKnotPoints(u(actuators_map.at(member)) <= 300/input_scale);
  // trajopt->AddConstraintToAllKnotPoints(u(actuators_map.at(member)) >= -300/input_scale);
  // }
  // Since the limit are the same, we don't need to loop over motor_names
  for (int i = 0; i < N; i++) {
    auto ui = trajopt->input(i);
    trajopt->AddBoundingBoxConstraint(
      VectorXd::Constant(n_u, -300 / input_scale),
      VectorXd::Constant(n_u, +300 / input_scale),
      ui);
  }

  // make sure it's left stance
  // trajopt->AddLinearConstraint(x0(positions_map.at("hip_pitch_left")) <=
  //                              x0(positions_map.at("hip_pitch_right")));


  // Testing - Cosntraint on the redundant contact force
  // cout << "This constraint doesn't work for heel to toe walking\n";
  // auto redundant_force_constraint = std::make_shared<RedundantForceConstraint>();
  // int counter = 0;
  // for (unsigned int i = 0; i < num_time_samples.size(); i++) {
  //   for (int j=0; j <  num_time_samples[i]; j++) {
  //     auto f = trajopt->force(i,j);
  //     if (dataset_list[i]->countConstraints() == 8) {
  //       // cout << i << ", " << j << endl;
  //       trajopt->AddConstraint(redundant_force_constraint,
  //           {f.segment(0,1), f.segment(3,1)});
  //     } else if (dataset_list[i]->countConstraints() == 14) {
  //       // cout << i << ", " << j << endl;
  //       trajopt->AddConstraint(redundant_force_constraint,
  //           {f.segment(0,1), f.segment(3,1)});
  //       trajopt->AddConstraint(redundant_force_constraint,
  //           {f.segment(6,1), f.segment(9,1)});
  //     }
  //   }
  //   counter += num_time_samples[i] - 1;
  // }



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
  // vector<VectorXd> v_seed = GetInitGuessForV(q_seed, duration / (N_ss - 1), plant);
  // MatrixXd S = 10 * MatrixXd::Identity(n_q + n_v, n_q + n_v);
  // for (int i = 0; i < N; i++) {
  //   auto xi = trajopt->state(i);
  //   VectorXd xi_seed(n_q + n_v);
  //   xi_seed << q_seed.at(i), v_seed.at(i);
  //   trajopt->AddQuadraticErrorCost(S, xi_seed, xi);
  // }






  // Create timesteps manually
  // vector<double> timestep_vector{0.0119019, 0.028669, 0};  // we don't use the third element, so set it to be any number
  // for (unsigned int i = 0; i < timestep_vector.size(); i++) {
  //   timestep_vector[i] /= time_scale;
  // }
  // DRAKE_DEMAND(timestep_vector.size() == num_time_samples.size());
  // VectorXd timesteps(N-1);
  // int counter = 0;
  // for (unsigned int i = 0; i < num_time_samples.size(); i++) {
  //   for (int j=0; j <  num_time_samples[i] - 1; j++) {
  //     int index = counter + j;
  //     timesteps(index) = timestep_vector[i];
  //   }
  //   counter += num_time_samples[i] - 1;
  // }
  // Fix the timestep sizes
  // counter = 0;
  // for (unsigned int i = 0; i < num_time_samples.size(); i++) {
  //   // for (int j=0; j <  num_time_samples[i]; j++) {
  //   //   int index = counter + j;
  //   //   trajopt->AddLinearConstraint(trajopt->timestep(index) == timesteps(index));
  //   // }

  //   if (counter >= N-1) break;
  //   trajopt->AddBoundingBoxConstraint(timestep_vector[counter], timestep_vector[counter],
  //                                     trajopt->timestep(counter));
  //   counter += num_time_samples[i] - 1;
  // }
  // make all timestep sizes the same
  int counter = 0;
  for (unsigned int i = 0; i < num_time_samples.size(); i++) {
    if (i > 0 && num_time_samples[i] > 1) {
      trajopt->AddLinearConstraint(trajopt->timestep(counter - 1) ==
                                   trajopt->timestep(counter));
    }
    counter += num_time_samples[i] - 1;
  }

  // add cost
  const MatrixXd Q = 10 * 12.5 *
                     omega_scale * omega_scale * MatrixXd::Identity(n_v, n_v);
  const MatrixXd R = 12.5 *
                     input_scale * input_scale * MatrixXd::Identity(n_u, n_u);
  // trajopt->AddRunningCost(x.tail(n_v).transpose()* Q * x.tail(n_v));
  // trajopt->AddRunningCost(x.segment(n_q,3).transpose() * 10.0 * x.segment(n_q,3));
  // trajopt->AddRunningCost(u.transpose()* R * u);

  // state cost
  // trajopt->AddQuadraticCost(Q * timesteps(0) / 2, VectorXd::Zero(n_x), x0.tail(n_v));
  // for (int i = 1; i <= N - 2; i++) {
  //   auto xi = trajopt->state(i);
  //   trajopt->AddQuadraticCost(Q * (timesteps(i-1)+timesteps(i))/2, VectorXd::Zero(n_x), xi.tail(n_v));
  // }
  // trajopt->AddQuadraticCost(Q * timesteps(N-2) / 2, VectorXd::Zero(n_x), xf.tail(n_v));
  // input cost
  // trajopt->AddQuadraticCost(R * timesteps(0) / 2, VectorXd::Zero(n_u), u0);
  // for (int i = 1; i <= N - 2; i++) {
  //   auto ui = trajopt->input(i);
  //   trajopt->AddQuadraticCost(R * (timesteps(i-1)+timesteps(i))/2, VectorXd::Zero(n_u), ui);
  // }
  // trajopt->AddQuadraticCost(R * timesteps(N-2) / 2, VectorXd::Zero(n_u), uf);

  // if all timesteps are the same
  double fixed_dt = duration / (N - 1);
  // trajopt->AddQuadraticCost(Q * fixed_dt / 2, VectorXd::Zero(n_v), x0.tail(n_v));
  trajopt->AddQuadraticCost(R * fixed_dt / 2, VectorXd::Zero(n_u), u0);
  for (int i = 1; i <= N - 2; i++) {
    auto xi = trajopt->state(i);
    auto ui = trajopt->input(i);
    // trajopt->AddQuadraticCost(Q * fixed_dt, VectorXd::Zero(n_v), xi.tail(n_v));
    trajopt->AddQuadraticCost(R * fixed_dt, VectorXd::Zero(n_u), ui);
  }
  // trajopt->AddQuadraticCost(Q * fixed_dt / 2, VectorXd::Zero(n_v), xf.tail(n_v));
  trajopt->AddQuadraticCost(R * fixed_dt / 2, VectorXd::Zero(n_u), uf);







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

      Vector3d pelvis_position(-dist / 2, 0, 1);
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
        xi_init << q_init.head(4) / quaternion_scale,
                q_init.tail(n_q - 4),
                VectorXd::Zero(n_v);
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
      vector<VectorXd> v_seed = GetInitGuessForV(q_seed, duration / (N_ss - 1),
                                plant);
      for (int i = 0; i < N; i++) {
        auto xi = trajopt->state(i);
        VectorXd xi_seed(n_q + n_v);
        if (i < N_ss) {
          xi_seed << q_seed.at(i).head(4) / quaternion_scale,
                  q_seed.at(i).tail(n_q - 4),
                  v_seed.at(i) / omega_scale;
        } else {
          xi_seed << q_seed.at(N_ss - 1).head(4) / quaternion_scale,
                  q_seed.at(N_ss - 1).tail(n_q - 4),
                  v_seed.at(N_ss - 1)  / omega_scale;
        }
        trajopt->SetInitialGuess(xi, xi_seed);
      }
      /*
      // Get approximated vdot by finite difference
      vector<VectorXd> vdot_approx = GetApproxVdot(v_seed, duration / (N - 1), plant);
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
      trajopt->SetInitialGuess(xi(0), 1 / quaternion_scale);
      trajopt->SetInitialGuess(xi(1), 0);
      trajopt->SetInitialGuess(xi(2), 0);
      trajopt->SetInitialGuess(xi(3), 0);
    }
  }

  cout << "Choose the best solver: " <<
       drake::solvers::ChooseBestSolver(*trajopt).name() << endl;

  cout << "Solving DIRCON\n\n";
  auto start = std::chrono::high_resolution_clock::now();
  const auto result = Solve(*trajopt, trajopt->initial_guess());
  SolutionResult solution_result = result.get_solution_result();
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  // trajopt->PrintSolution();
  for (int i = 0; i < 100; i++) {cout << '\a';}  // making noise to notify
  cout << "\n" << to_string(solution_result) << endl;
  cout << "Solve time:" << elapsed.count() << std::endl;
  cout << "Cost:" << result.get_optimal_cost() << std::endl;

  // Check which solver we are using
  cout << "Solver: " << result.get_solver_id().name() << endl;

  // Testing - check if the nonilnear constraints are all satisfied
  bool constraint_satisfied = solvers::CheckGenericConstraints(*trajopt, result,
                              1e-5);
  cout << "constraint_satisfied = " << constraint_satisfied << endl;

  // store the solution of the decision variable
  VectorXd z = result.GetSolution(trajopt->decision_variables());
  writeCSV(data_directory + output_prefix + string("z.csv"), z);
  for (int i = 0; i < z.size(); i++) {
    cout << trajopt->decision_variables()[i] << ", " << z[i] << endl;
  }
  cout << endl;

  // store the time, state, and input at knot points
  VectorXd time_at_knots = trajopt->GetSampleTimes(result);
  MatrixXd state_at_knots = trajopt->GetStateSamples(result);
  MatrixXd input_at_knots = trajopt->GetInputSamples(result);
  state_at_knots.col(N - 1) = result.GetSolution(xf);
  time_at_knots *= time_scale;
  state_at_knots
      << state_at_knots.block(0, 0, 4, state_at_knots.cols()) * quaternion_scale,
      state_at_knots.block(4, 0, n_q - 4, state_at_knots.cols()),
      state_at_knots.block(n_q, 0, n_v, state_at_knots.cols()) * omega_scale;
  cout << "you'll need to update state_at_knots if it's multiple modes\n";
  input_at_knots *= input_scale;
  writeCSV(data_directory + string("t_i.csv"), time_at_knots);
  writeCSV(data_directory + string("x_i.csv"), state_at_knots);
  writeCSV(data_directory + string("u_i.csv"), input_at_knots);
  cout << "time_at_knots = \n" << time_at_knots << "\n";
  cout << "state_at_knots = \n" << state_at_knots << "\n";
  cout << "state_at_knots.size() = " << state_at_knots.size() << endl;
  cout << "input_at_knots = \n" << input_at_knots << "\n";

  // Also store lambda. We might need to look at it in the future!
  // (save it so we don't need to rerun)
  std::ofstream ofile;
  ofile.open("examples/Cassie/trajopt_data/lambda.txt",
             std::ofstream::out);
  cout << "lambda_sol = \n";
  for (unsigned int mode = 0; mode < num_time_samples.size(); mode++) {
    for (int index = 0; index < num_time_samples[mode]; index++) {
      auto lambdai = trajopt->force(mode, index);
      cout << result.GetSolution(lambdai).transpose() * force_scale << endl;
      ofile << result.GetSolution(lambdai).transpose() * force_scale << endl;
    }
  }
  ofile.close();

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

  double duration = FLAGS_duration; //0.5
  int iter = FLAGS_max_iter;
  string data_directory = "examples/Cassie/trajopt_data/";
  string init_file = FLAGS_init_file;
  // string init_file = "testing_z.csv";
  string output_prefix = "";

  dairlib::DoMain(FLAGS_stride_length, FLAGS_ground_incline,
                  duration, iter,
                  data_directory, init_file, output_prefix,
                  FLAGS_omega_scale,
                  FLAGS_input_scale,
                  FLAGS_force_scale,
                  FLAGS_time_scale,
                  FLAGS_quaternion_scale,
                  FLAGS_tol);
}

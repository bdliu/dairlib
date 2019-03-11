#include "examples/Goldilocks_models/traj_opt_given_weigths.h"
#include "examples/Goldilocks_models/goldilocks_model_traj_opt.h"

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

#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/constraint.h"
#include "drake/solvers/solve.h"
#include "systems/trajectory_optimization/dircon_util.h"
#include "systems/trajectory_optimization/dircon_position_data.h"
#include "systems/trajectory_optimization/dircon_kinematic_data_set.h"
#include "systems/trajectory_optimization/dircon_opt_constraints.h"
#include "systems/trajectory_optimization/hybrid_dircon.h"

#include "multibody/multibody_utils.h"
#include "multibody/visualization_utils.h"

#include "systems/goldilocks_models/symbolic_manifold.h"
#include "systems/goldilocks_models/file_utils.h"

using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Matrix3Xd;
using drake::VectorX;
using drake::systems::trajectory_optimization::MultipleShooting;
using drake::trajectories::PiecewisePolynomial;
using drake::solvers::Binding;
using drake::solvers::Constraint;
using drake::solvers::VectorXDecisionVariable;
using drake::solvers::MatrixXDecisionVariable;
using drake::symbolic::Variable;
using drake::symbolic::Expression;
using std::shared_ptr;
using std::cout;
using std::endl;
using std::string;
using std::map;

using drake::multibody::MultibodyPlant;
using drake::geometry::SceneGraph;
using drake::multibody::Body;
using drake::multibody::Parser;
using drake::systems::rendering::MultibodyPositionToGeometryPose;

using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
using drake::solvers::SolutionResult;

// using Isometry3 = Eigen::Transform<Scalar, 3, Eigen::Isometry>

namespace dairlib {
namespace goldilocks_models {

using systems::trajectory_optimization::HybridDircon;
using systems::trajectory_optimization::DirconDynamicConstraint;
using systems::trajectory_optimization::DirconKinematicConstraint;
using systems::trajectory_optimization::DirconOptions;
using systems::trajectory_optimization::DirconKinConstraintType;
using systems::SubvectorPassThrough;

void trajOptGivenWeights(int n_z, int n_zDot, int n_featureZ, int n_featureZDot,
                      Eigen::VectorXd & thetaZ, Eigen::VectorXd & thetaZDot,
                      double stride_length, double duration, int max_iter,
                      string directory,
                      string init_file, std::string output_prefix,
                      vector<VectorXd> & w_sol_vec,
                      vector<MatrixXd> & A_vec, vector<MatrixXd> & H_vec,
                      vector<VectorXd> & y_vec,
                      vector<VectorXd> & lb_vec, vector<VectorXd> & ub_vec,
                      vector<VectorXd> & b_vec,
                      vector<MatrixXd> & B_vec) {
  drake::systems::DiagramBuilder<double> builder;
  MultibodyPlant<double> plant;
  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  Parser parser(&plant, &scene_graph);

  std::string full_name =
    FindResourceOrThrow("examples/Goldilocks_models/PlanarWalkerWithTorso.urdf");
  parser.AddModelFromFile(full_name);
  plant.AddForceElement<drake::multibody::UniformGravityFieldElement>(
    -9.81 * Eigen::Vector3d::UnitZ());
  plant.WeldFrames(
    plant.world_frame(), plant.GetFrameByName("base"),
    drake::math::RigidTransform<double>(Vector3d::Zero()).GetAsIsometry3());
  plant.Finalize();

  // Create autoDiff version of the plant
  MultibodyPlant<AutoDiffXd> plant_autoDiff(plant);


  map<string, int> positions_map = multibody::makeNameToPositionsMap(plant);
  map<string, int> velocities_map = multibody::makeNameToVelocitiesMap(
                                      plant);
  map<string, int> actuators_map = multibody::makeNameToActuatorsMap(plant);
  // for (auto const& element : positions_map)
  //   cout << element.first << " = " << element.second << endl;
  // std::cout << "\n";
  // for (auto const& element : velocities_map)
  //   cout << element.first << " = " << element.second << endl;
  // std::cout << "\n";
  // for (auto const& element : actuators_map)
  //   cout << element.first << " = " << element.second << endl;
  // std::cout << "\n";


  int n_q = plant.num_positions();
  // int n_v = plant.num_velocities();
  // int n_x = n_q + n_v;
  // int n_u = plant.num_actuators();
  // std::cout<<"n_x = "<<n_x<<"\n";
  // std::cout<<"n_u = "<<n_u<<"\n";

  const Body<double>& left_lower_leg = plant.GetBodyByName("left_lower_leg");
  const Body<double>& right_lower_leg =
    plant.GetBodyByName("right_lower_leg");

  Vector3d pt;
  pt << 0, 0, -.5;
  bool isXZ = true;

  auto leftFootConstraint = DirconPositionData<double>(plant, left_lower_leg,
                            pt, isXZ);
  auto rightFootConstraint = DirconPositionData<double>(plant,
                             right_lower_leg,
                             pt, isXZ);

  Vector3d normal;
  normal << 0, 0, 1;
  double mu = 1;
  leftFootConstraint.addFixedNormalFrictionConstraints(normal, mu);
  rightFootConstraint.addFixedNormalFrictionConstraints(normal, mu);
  // std::cout<<leftFootConstraint.getLength()<<"\n"; //2 dim. I guess the contact point constraint in the x and z direction

  std::vector<DirconKinematicData<double>*> leftConstraints;
  leftConstraints.push_back(&leftFootConstraint);
  auto leftDataSet = DirconKinematicDataSet<double>(plant, &leftConstraints);

  std::vector<DirconKinematicData<double>*> rightConstraints;
  rightConstraints.push_back(&rightFootConstraint);
  auto rightDataSet = DirconKinematicDataSet<double>(plant,
                      &rightConstraints);

  auto leftOptions = DirconOptions(leftDataSet.countConstraints());
  leftOptions.setConstraintRelative(0,
                                    true); //TODO: ask what is relative constraint here?
  // std::cout<<"leftDataSet.countConstraints() = "<<leftDataSet.countConstraints()<<"\n";

  auto rightOptions = DirconOptions(rightDataSet.countConstraints());
  rightOptions.setConstraintRelative(0, true);

  // Stated in the MultipleShooting class:
  // This class assumes that there are a fixed number (N) time steps/samples,
  // and that the trajectory is discretized into timesteps h (N-1 of these),
  // state x (N of these), and control input u (N of these).
  std::vector<int> num_time_samples;
  num_time_samples.push_back(20); // First mode (20 sample points)
  num_time_samples.push_back(1);  // Second mode (1 sample point)
  std::vector<double> min_dt;
  min_dt.push_back(
    .01);   // bound for time difference between adjacent samples in the first mode // See HybridDircon constructor
  min_dt.push_back(
    .01);   // bound for time difference between adjacent samples in the second mode
  std::vector<double> max_dt;
  max_dt.push_back(.3);
  max_dt.push_back(.3);

  int N = 0;
  for (uint i = 0; i < num_time_samples.size(); i++)
    N += num_time_samples[i];
  N -= num_time_samples.size() - 1; //Overlaps between modes
  // std::cout<<"N = "<<N<<"\n";

  std::vector<DirconKinematicDataSet<double>*> dataset_list;
  dataset_list.push_back(&leftDataSet);
  dataset_list.push_back(&rightDataSet);

  std::vector<DirconOptions> options_list;
  options_list.push_back(leftOptions);
  options_list.push_back(rightOptions);

  auto trajopt = std::make_unique<HybridDircon<double>>(plant,
                 num_time_samples, min_dt, max_dt, dataset_list, options_list);

  // trajopt->AddDurationBounds(duration, duration); // You can comment this out to not put any constraint on the time

  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Print file", "snopt.out");
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Major iterations limit", max_iter);
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(), "Verify level",
                           0);


  // Periodicity constraints
  auto x0 = trajopt->initial_state();
  // auto xf = trajopt->final_state();
  auto xf = trajopt->state_vars_by_mode(num_time_samples.size() - 1,
                                        num_time_samples[num_time_samples.size() - 1] - 1);

  //Careful! if you have a string typo, the code still runs and the mapped value will be 0.
  trajopt->AddLinearConstraint(x0(positions_map.at("planar_z")) == xf(
                                 positions_map.at("planar_z")));
  trajopt->AddLinearConstraint(x0(positions_map.at("planar_roty")) == xf(
                                 positions_map.at("planar_roty")));
  trajopt->AddLinearConstraint(x0(positions_map.at("left_hip_pin")) == xf(
                                 positions_map.at("right_hip_pin")));
  trajopt->AddLinearConstraint(x0(positions_map.at("left_knee_pin")) == xf(
                                 positions_map.at("right_knee_pin")));
  trajopt->AddLinearConstraint(x0(positions_map.at("right_hip_pin")) == xf(
                                 positions_map.at("left_hip_pin")));
  trajopt->AddLinearConstraint(x0(positions_map.at("right_knee_pin")) == xf(
                                 positions_map.at("left_knee_pin")));

  trajopt->AddLinearConstraint(x0(n_q + velocities_map.at("planar_xdot"))
                           == xf(n_q + velocities_map.at("planar_xdot")));
  trajopt->AddLinearConstraint(x0(n_q + velocities_map.at("planar_zdot"))
                           == xf(n_q + velocities_map.at("planar_zdot")));
  trajopt->AddLinearConstraint(x0(n_q + velocities_map.at("planar_rotydot"))
                           == xf(n_q + velocities_map.at("planar_rotydot")));
  trajopt->AddLinearConstraint(x0(n_q + velocities_map.at("left_hip_pindot"))
                           == xf(n_q + velocities_map.at("right_hip_pindot")));
  trajopt->AddLinearConstraint(x0(n_q + velocities_map.at("left_knee_pindot"))
                           == xf(n_q + velocities_map.at("right_knee_pindot")));
  trajopt->AddLinearConstraint(x0(n_q + velocities_map.at("right_hip_pindot"))
                           == xf(n_q + velocities_map.at("left_hip_pindot")));
  trajopt->AddLinearConstraint(x0(n_q + velocities_map.at("right_knee_pindot"))
                           == xf(n_q + velocities_map.at("left_knee_pindot")));

  // u periodic constraint
  auto u0 = trajopt->input(0);
  auto uf = trajopt->input(N - 1);
  trajopt->AddLinearConstraint(u0(actuators_map.at("left_hip_torque")) == uf(
                                 actuators_map.at("right_hip_torque")));
  trajopt->AddLinearConstraint(u0(actuators_map.at("right_hip_torque")) == uf(
                                 actuators_map.at("left_hip_torque")));
  trajopt->AddLinearConstraint(u0(actuators_map.at("left_knee_torque")) == uf(
                                 actuators_map.at("right_knee_torque")));
  trajopt->AddLinearConstraint(u0(actuators_map.at("right_knee_torque")) == uf(
                                 actuators_map.at("left_knee_torque")));

  // Knee joint limits
  auto x = trajopt->state();
  trajopt->AddConstraintToAllKnotPoints(x(positions_map.at("left_knee_pin")) >=
                                        5.0 / 180.0 * M_PI);
  trajopt->AddConstraintToAllKnotPoints(x(positions_map.at("right_knee_pin")) >=
                                        5.0 / 180.0 * M_PI);
  trajopt->AddConstraintToAllKnotPoints(x(positions_map.at("left_knee_pin")) <=
                                        M_PI / 2.0);
  trajopt->AddConstraintToAllKnotPoints(x(positions_map.at("right_knee_pin")) <=
                                        M_PI / 2.0);

  // hip joint limits
  trajopt->AddConstraintToAllKnotPoints(x(positions_map.at("left_hip_pin")) >=
                                        -M_PI / 2.0);
  trajopt->AddConstraintToAllKnotPoints(x(positions_map.at("right_hip_pin")) >=
                                        -M_PI / 2.0);
  trajopt->AddConstraintToAllKnotPoints(x(positions_map.at("left_hip_pin")) <=
                                        M_PI / 2.0);
  trajopt->AddConstraintToAllKnotPoints(x(positions_map.at("right_hip_pin")) <=
                                        M_PI / 2.0);

  // x-distance constraint constraints
  trajopt->AddLinearConstraint(x0(positions_map.at("planar_x")) == 0);
  trajopt->AddLinearConstraint(xf(positions_map.at("planar_x")) ==
                               stride_length);

  // make sure it's left stance
  trajopt->AddLinearConstraint(x0(positions_map.at("left_hip_pin")) <=
                                   x0(positions_map.at("right_hip_pin")));

  // add cost
  const double R = 10;  // Cost on input effort
  auto u = trajopt->input();
  trajopt->AddRunningCost(u.transpose()*R * u);
  MatrixXd Q = MatrixXd::Zero(2 * n_q, 2 * n_q);
  for (int i = 0; i < n_q; i++) {
    Q(i + n_q, i + n_q) = 10;
  }
  trajopt->AddRunningCost(x.transpose()*Q * x);

  // initial guess if the file exists
  if (!init_file.empty()) {
    MatrixXd w0 = readCSV(directory + init_file);
    trajopt->SetInitialGuessForAllVariables(w0);
  }

  // Move the trajectory optmization problem into GoldilcocksModelTrajOpt
  // where we add the constraints for reduced order model
  GoldilcocksModelTrajOpt gm_traj_opt(
      n_z, n_zDot, n_featureZ, n_featureZDot, thetaZ, thetaZDot,
      std::move(trajopt), &plant_autoDiff, num_time_samples);

  std::cout << "Solving DIRCON (based on MultipleShooting)\n";
  auto start = std::chrono::high_resolution_clock::now();
  const MathematicalProgramResult result = Solve(
    *gm_traj_opt.dircon, gm_traj_opt.dircon->initial_guess());
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Solve time:" << elapsed.count() << std::endl;
  SolutionResult solution_result = result.get_solution_result();
  std::cout << solution_result << std::endl;
  std::cout << "Cost:" << result.get_optimal_cost() << std::endl;

  // The following line gives seg fault
  // systems::trajectory_optimization::checkConstraints(trajopt.get(), result);




  // Testing: print out the timesteps
  // for(int i = 0; i<N-1 ; i++){
  //   auto h_i = gm_traj_opt.dircon->timestep(i);
  //   VectorXd h_i_sol = gm_traj_opt.dircon->GetSolution(h_i);
  //   cout << "h_"<< i <<"_sol = " << h_i_sol << endl;
  // }
  auto h_0 = gm_traj_opt.dircon->timestep(0);
  VectorXd h_0_sol = result.GetSolution(h_0);
  cout << "timestep = " << h_0_sol << endl;

  cout << endl;
  // Testing: print out the vertical pos
  for(int i = 0; i<N ; i++){
    auto x_i = gm_traj_opt.dircon->state(i);
    VectorXd x_i_sol = result.GetSolution(x_i);
    cout << "x1_"<< i <<"_sol = " << x_i_sol(1) << endl;
  }

  for(int i = 0; i<N ; i++){
      auto z_k = gm_traj_opt.reduced_model_state(i, n_z);
      VectorXd z_k_sol = result.GetSolution(z_k);
      cout << "z_"<< i <<"_sol = " << z_k_sol.transpose() << endl;
  }






  // Get the solution of all the decision variable
  VectorXd w_sol = result.GetSolution(
                 gm_traj_opt.dircon->decision_variables());

  // Assume theta is fixed. Get the linear approximation of the cosntraints and
  // second order approximation of the cost.
  MatrixXd A, H;
  VectorXd y, lb, ub, b;
  systems::trajectory_optimization::linearizeConstraints(
    gm_traj_opt.dircon.get(), w_sol, y, A, lb, ub);
  double costval = systems::trajectory_optimization::secondOrderCost(
    gm_traj_opt.dircon.get(), w_sol, H, b);

  // Get matrix B (~get feature vectors)
  int n_thetaZ = thetaZ.size();
  int n_thetaZDot = thetaZDot.size();
  MatrixXd B = MatrixXd::Zero(A.rows(), n_thetaZ + n_thetaZDot);
  ///////////////////////// Kinematics Constraints /////////////////////////////
  // Get the row index of B matrix where kinematics constraint starts
  VectorXd ind = systems::trajectory_optimization::getConstraintRows(
    gm_traj_opt.dircon.get(),
    gm_traj_opt.kinematics_constraint_bindings[0]);
  for (int i = 0; i < N; i++) {
    // Get the gradient value first
    VectorXd xi = result.GetSolution(
        gm_traj_opt.dircon->state(i));
    VectorXd kin_gradient =
        gm_traj_opt.kinematics_constraint->getGradientWrtTheta(xi);

    // Fill in B matrix
    for (int k = 0; k < n_z; k++) {
      for (int j = 0; j < kin_gradient.size(); j++) {
        B(ind(0) + i*n_z + k, k*kin_gradient.size() + j) = kin_gradient(j);
        // cout << "ind(0) + i*n_z + k = " << ind(0) + i*n_z + k << endl;
      }
    }
  }
  /////////////////////////// Dynamics Constraints /////////////////////////////
  // Get the row index of B matrix where kinematics constraint starts
  ind = systems::trajectory_optimization::getConstraintRows(
    gm_traj_opt.dircon.get(),
    gm_traj_opt.dynamics_constraint_bindings[0]);
  int N_accum = 0;
  int p = 0; // because we skip the last segment of each mode, so "i" doesn't count from 1 to ...
  for (unsigned int l = 0; l < num_time_samples.size() ; l++) {
    for (int m = 0; m < num_time_samples[l]-2 ; m++) {
      int i = N_accum + m;
      // Get the gradient value first
      auto z_i = gm_traj_opt.reduced_model_state(i, n_z);
      auto z_iplus1 = gm_traj_opt.reduced_model_state(i + 1, n_z);
      auto h_btwn_knot_i_iplus1 = gm_traj_opt.dircon->timestep(i);
      VectorXd z_i_sol = result.GetSolution(z_i);
      VectorXd z_iplus1_sol = result.GetSolution(z_iplus1);
      VectorXd h = result.GetSolution(h_btwn_knot_i_iplus1);

      VectorXd dyn_gradient =
          gm_traj_opt.dynamics_constraint->getGradientWrtTheta(
            z_i_sol, z_iplus1_sol, h);
      // cout<< "("<< l<< ", "<< m<<  "): dyn_gradient = " << dyn_gradient.transpose() << endl;

      // Fill in B matrix
      for (int k = 0; k < n_zDot; k++) {
        for (int j = 0; j < dyn_gradient.size(); j++) {
          B(ind(0) + p*n_zDot + k, n_thetaZ + k*dyn_gradient.size() + j) =
            dyn_gradient(j);
          // cout << "ind(0) + p*n_zDot + k = " << ind(0) + p*n_zDot + k << endl;
        }
      }
      p++;
    }
    N_accum += num_time_samples[l];
    N_accum -= 1;  // due to overlaps between modes
  }

  // Push the solution to the vector
  w_sol_vec.push_back(w_sol);
  A_vec.push_back(A);
  H_vec.push_back(H);
  lb_vec.push_back(lb);
  ub_vec.push_back(ub);
  y_vec.push_back(y);
  b_vec.push_back(b);
  B_vec.push_back(B);

  // Store the vectors and matrices
  // string batch_prefix = std::to_string(iter-1) + "_" + std::to_string(batch) + "_";
  // string iter_prefix = std::to_string(iter-1) + "_";
  writeCSV(directory + output_prefix + string("w.csv"), w_sol);
  writeCSV(directory + output_prefix + string("A.csv"), A);
  writeCSV(directory + output_prefix + string("y.csv"), y);
  writeCSV(directory + output_prefix + string("lb.csv"), lb);
  writeCSV(directory + output_prefix + string("ub.csv"), ub);
  writeCSV(directory + output_prefix + string("H.csv"), H);
  writeCSV(directory + output_prefix + string("b.csv"), b);
  writeCSV(directory + output_prefix + string("B.csv"), B);

  // Store the time, state, and input at knot points
  VectorXd time_at_knots = gm_traj_opt.dircon->GetSampleTimes(result);
  MatrixXd state_at_knots = gm_traj_opt.dircon->GetStateSamples(result);
  MatrixXd input_at_knots = gm_traj_opt.dircon->GetInputSamples(result);
  writeCSV(directory + output_prefix + string("time_at_knots.csv"), time_at_knots);
  writeCSV(directory + output_prefix + string("state_at_knots.csv"), state_at_knots);
  writeCSV(directory + output_prefix + string("input_at_knots.csv"), input_at_knots);

  /*// visualizer
  const PiecewisePolynomial<double> pp_xtraj =
    gm_traj_opt.dircon->ReconstructStateTrajectory(result);
  multibody::connectTrajectoryVisualizer(&plant, &builder, &scene_graph,
                                         pp_xtraj);
  auto diagram = builder.Build();
  while (true) {
    drake::systems::Simulator<double> simulator(*diagram);
    simulator.set_target_realtime_rate(.1);
    simulator.Initialize();
    simulator.StepTo(pp_xtraj.end_time());
  }*/

  return ;
}

}  // namespace goldilocks_models
}  // namespace dairlib


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
                         vector<MatrixXd> & B_vec,
                         const double & Q_double, const double & R_double,
                         double epsilon) {
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
  int n_v = plant.num_velocities();
  int n_x = n_q + n_v;
  int n_u = plant.num_actuators();
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

  // You can comment this out to not put any constraint on the time
  // However, we need it now, since we add the running cost by hand
  trajopt->AddDurationBounds(duration, duration);

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

  // Add cost
  MatrixXd R = MatrixXd::Identity(n_u, n_u);
  MatrixXd Q = MatrixXd::Zero(n_x, n_x);
  MatrixXd I_x = MatrixXd::Identity(n_x, n_x);
  for (int i = 0; i < n_v; i++) {
    Q(i + n_q, i + n_q) = Q_double;
  }
  // Don't use AddRunningCost, cause it makes cost Hessian to be indefinite
  // I'll fix the timestep and add cost myself
  /*auto u = trajopt->input();
  trajopt->AddRunningCost(u.transpose()*R * u);
  trajopt->AddRunningCost(x.transpose()*Q * x);*/
  // Add running cost by hand (Trapezoidal integration):
  double timestep = duration / (N - 1);
  trajopt->AddQuadraticCost(Q * timestep / 2, VectorXd::Zero(n_x), x0);
  trajopt->AddQuadraticCost(R * timestep / 2, VectorXd::Zero(n_u), u0);
  for (int i = 1; i <= N - 2; i++) {
    auto ui = trajopt->input(i);
    auto xi = trajopt->state(i);
    trajopt->AddQuadraticCost(Q * timestep, VectorXd::Zero(n_x), xi);
    trajopt->AddQuadraticCost(R * timestep, VectorXd::Zero(n_u), ui);
  }
  trajopt->AddQuadraticCost(Q * timestep / 2, VectorXd::Zero(n_x), xf);
  trajopt->AddQuadraticCost(R * timestep / 2, VectorXd::Zero(n_u), uf);
  // Add regularization term here so that hessian is pd (for outer loop), so
  // that we can use schur complement method
  // Actually, Hessian still has zero eigen value because of h_var and z_var
  /*trajopt->AddQuadraticCost(epsilon*I_x*timestep/2,VectorXd::Zero(n_x),x0);
  for(int i=1; i<=N-2; i++){
    auto xi = trajopt->state(i);
    trajopt->AddQuadraticCost(epsilon*I_x*timestep,VectorXd::Zero(n_x),xi);
  }
  trajopt->AddQuadraticCost(epsilon*I_x*timestep/2,VectorXd::Zero(n_x),xf);*/


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

  // Add regularization term here so that hessian is pd (for outer loop), so
  // that we can use schur complement method
  // TODO(yminchen): should I add to all decision variable? or just state?
  auto w = gm_traj_opt.dircon->decision_variables();
  for (int i = 0; i < w.size(); i++)
    gm_traj_opt.dircon->AddQuadraticCost(epsilon * w(i)*w(i));


  cout << "Solving DIRCON (based on MultipleShooting)\n";
  auto start = std::chrono::high_resolution_clock::now();
  const MathematicalProgramResult result = Solve(
        *gm_traj_opt.dircon, gm_traj_opt.dircon->initial_guess());
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  cout << "Solve time:" << elapsed.count() << endl;
  SolutionResult solution_result = result.get_solution_result();
  cout << solution_result << endl;
  cout << "Cost:" << result.get_optimal_cost() << endl << endl;

  // The following line gives seg fault
  // systems::trajectory_optimization::checkConstraints(trajopt.get(), result);




  // Testing: print out the timesteps
  // for(int i = 0; i<N-1 ; i++){
  //   auto h_i = gm_traj_opt.dircon->timestep(i);
  //   VectorXd h_i_sol = gm_traj_opt.dircon->GetSolution(h_i);
  //   cout << "h_"<< i <<"_sol = " << h_i_sol << endl;
  // }
  /*auto h_0 = gm_traj_opt.dircon->timestep(0);
  VectorXd h_0_sol = result.GetSolution(h_0);
  cout << "timestep = " << h_0_sol << endl << endl;*/

  // Testing: print out the vertical pos
  /*for(int i = 0; i<N ; i++){
    auto x_i = gm_traj_opt.dircon->state(i);
    VectorXd x_i_sol = result.GetSolution(x_i);
    cout << "x1_"<< i <<"_sol = " << x_i_sol(1) << endl;
  }

  for(int i = 0; i<N ; i++){
      auto z_k = gm_traj_opt.reduced_model_state(i, n_z);
      VectorXd z_k_sol = result.GetSolution(z_k);
      cout << "z_"<< i <<"_sol = " << z_k_sol.transpose() << endl;
  }*/

  // Testing: see what are the decision varaibles
  // cout << "\nAll decision variable:\n"
  //   << gm_traj_opt.dircon->decision_variables() << endl;



  // Get the solution of all the decision variable
  VectorXd w_sol = result.GetSolution(
                     gm_traj_opt.dircon->decision_variables());

  // Assume theta is fixed. Get the linear approximation of the cosntraints and
  // second order approximation of the cost.
  MatrixXd A, H;
  VectorXd y, lb, ub, b;
  double c;
  systems::trajectory_optimization::linearizeConstraints(
    gm_traj_opt.dircon.get(), w_sol, y, A, lb, ub);
  c = systems::trajectory_optimization::secondOrderCost(
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
        B(ind(0) + i * n_z + k, k * kin_gradient.size() + j) = kin_gradient(j);
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
    for (int m = 0; m < num_time_samples[l] - 2 ; m++) {
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
      for (int k = 0; k < n_zDot / 2; k++) {
        for (int j = 0; j < dyn_gradient.size(); j++) {
          B(ind(0) + p * n_zDot + k, n_thetaZ + k * dyn_gradient.size() + j) =
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




  cout << "\ncheck if H is diagonal: \n";
  MatrixXd H_test = H;
  int nw = H_test.rows();
  for (int i = 0; i < nw; i++) {
    H_test(i, i) = 0;
  }
  if (VectorXd::Ones(nw).transpose()*H_test * VectorXd::Ones(nw) == 0)
    cout << "H is diagonal" << endl;

  cout << "checking b\n";
  cout << "b.norm() = " << b.norm() << endl;



  cout << "norm of y = " << y.norm() << endl;





  // Get the index of the rows of active constraints
  vector<int> active_eq_row_idx;
  vector<int> ub_active_ineq_row_idx;
  vector<int> lb_active_ineq_row_idx;
  double tol = 1e-4; //1e-4
  for (int i = 0; i < y.rows(); i++) {
    if (ub(i) == lb(i))
      active_eq_row_idx.push_back(i);
    else if (y(i) >= ub(i) - tol)
      ub_active_ineq_row_idx.push_back(i);
    else if (y(i) <= lb(i) + tol)
      lb_active_ineq_row_idx.push_back(i);
  }
  unsigned int n_ae = active_eq_row_idx.size();
  unsigned int n_aub = ub_active_ineq_row_idx.size();
  unsigned int n_alb = lb_active_ineq_row_idx.size();
  cout << "n_ae = " << n_ae << endl;
  cout << "n_aub = " << n_aub << endl;
  cout << "n_alb = " << n_alb << endl;




  cout << "\nRun traj opt to check if your quadratic approximation is correct\n";
  // int nl_i = A.rows();
  int nw_i = A.cols();
  MathematicalProgram quadprog;
  auto dw = quadprog.NewContinuousVariables(nw_i, "dw");
  quadprog.AddLinearConstraint( A,
                                lb - y,
                                ub - y,
                                dw);
  quadprog.AddQuadraticCost(H, b, dw);
  const auto result2 = Solve(quadprog);
  auto solution_result2 = result2.get_solution_result();
  cout << solution_result2 << endl;
  cout << "Cost:" << result2.get_optimal_cost() << endl;
  VectorXd dw_sol = result2.GetSolution(quadprog.decision_variables());
  cout << "w_sol norm:" << dw_sol.norm() << endl;
  cout << "Finished traj opt\n\n";

  // Plug back and check the cost and constraints of nonlinear programming
  double eps = 1e-1;
  unsigned int n_show = 10;  // number of rows of constraints you want to show
  // cost
  cout << "checking the cost of the original nonlinear programming and the approximated quadratic programming\n";
  for (int i = 0; i < 10 ; i++) {
    VectorXd w_sol_test = w_sol + i * eps * dw_sol;
    MatrixXd H2;
    VectorXd b2;
    double c_nonlinear;
    c_nonlinear = systems::trajectory_optimization::secondOrderCost(
                    gm_traj_opt.dircon.get(), w_sol_test, H2, b2) - c;
    cout << "i = " << i << endl;
    cout << "  c_nonlinear = " << c_nonlinear << endl;
    VectorXd dw_sol_test = i * eps * dw_sol;
    double c_aquadprog = 0.5 * dw_sol_test.transpose() * H * dw_sol_test + b.dot(
                           dw_sol_test) + c - c;
    cout << "  c_aquadprog = " << c_aquadprog << endl;
    cout << "  c_aquadprog - c_nonlinear = " << c_aquadprog - c_nonlinear << endl;
  }
  // constraint
  if (n_ae) {
    cout << "\nchecking the equality constraints of the original nonlinear programming and the approximated quadratic programming\n";
    // pick constraint violation row index
    std::vector<int> constraint_vio_row_idx;
    for (unsigned int i = 0; i < 10 ; i++) {
      VectorXd w_sol_test = w_sol + i * eps * dw_sol;
      MatrixXd A2;
      VectorXd y2, lb2, ub2;
      systems::trajectory_optimization::linearizeConstraints(
        gm_traj_opt.dircon.get(), w_sol_test, y2, A2, lb2, ub2);
      unsigned int k = 0;
      for (unsigned int j = 0; j < n_ae; j++) {
        double violation = y2(active_eq_row_idx[j]) - ub(active_eq_row_idx[j]);
        if (abs(violation) > 1e-8) {
          constraint_vio_row_idx.push_back(j);
          k++;
          if (k == n_show)
            break;
        }
        if (i == n_show-1 && j == n_ae - 1 && k < n_show) {
          cout << "There are only " << k << " # of violations\n";
        }
      }
      if(constraint_vio_row_idx.size()>=n_show)
        break;
      else if (i != 10-1)
        constraint_vio_row_idx.clear();
    }
    cout << "  Row index of violation = ";
    for(int j : constraint_vio_row_idx){
      cout << j << ", ";
    }
    cout << endl;
    // evaluate the chosen rows
    for (unsigned int i = 0; i < 10 ; i++) {
      cout << "i = " << i << endl;
      VectorXd w_sol_test = w_sol + i * eps * dw_sol;
      MatrixXd A2;
      VectorXd y2, lb2, ub2;
      systems::trajectory_optimization::linearizeConstraints(
        gm_traj_opt.dircon.get(), w_sol_test, y2, A2, lb2, ub2);
      cout << "  nonlinear_constraint_val = ";
      for(int j : constraint_vio_row_idx){
        double violation = y2(active_eq_row_idx[j]) - ub(active_eq_row_idx[j]);
        cout << violation << ", ";
      }
      cout << endl;
    }
  }
  if (n_aub) {
    cout << "\nchecking the inequality constraints (active upper bound) of the original nonlinear programming and the approximated quadratic programming\n";
    for (unsigned int i = 0; i < 10 ; i++) {
      VectorXd w_sol_test = w_sol + i * eps * dw_sol;
      MatrixXd A2;
      VectorXd y2, lb2, ub2;
      systems::trajectory_optimization::linearizeConstraints(
        gm_traj_opt.dircon.get(), w_sol_test, y2, A2, lb2, ub2);
      VectorXd nonlinear_constraint_val = VectorXd::Zero(n_show);
      unsigned int k = 0;
      for (unsigned int j = 0; j < n_aub; j++) {
        double violation =
            y2(ub_active_ineq_row_idx[j]) - ub(ub_active_ineq_row_idx[j]);
        if (violation > 1e-8) {
          nonlinear_constraint_val(k) = violation;
          k++;
          if (k == n_show)
            break;
        }
        if (j == n_aub - 1 && k < n_show) {
          cout << "There are only " << k << " # of violations\n";
        }
      }
      cout << "  nonlinear_constraint_val = "
          << nonlinear_constraint_val.transpose() << endl;
    }
  }
  if (n_alb) {
    cout << "\nchecking the inequality constraints (active lower bound) of the original nonlinear programming and the approximated quadratic programming\n";
    for (unsigned int i = 0; i < 10 ; i++) {
      VectorXd w_sol_test = w_sol + i * eps * dw_sol;
      MatrixXd A2;
      VectorXd y2, lb2, ub2;
      systems::trajectory_optimization::linearizeConstraints(
        gm_traj_opt.dircon.get(), w_sol_test, y2, A2, lb2, ub2);
      VectorXd nonlinear_constraint_val = VectorXd::Zero(n_show);
      unsigned int k = 0;
      for (unsigned int j = 0; j < n_alb; j++) {
        double violation =
            y2(lb_active_ineq_row_idx[j]) - lb(lb_active_ineq_row_idx[j]);
        if (violation < - 1e-8) {
          nonlinear_constraint_val(k) = violation;
          k++;
          if (k == n_show)
            break;
        }
        if (j == n_alb - 1 && k < n_show) {
          cout << "There are only " << k << " # of violations\n";
        }
      }
      cout << "  nonlinear_constraint_val = "
          << nonlinear_constraint_val.transpose() << endl;
    }
  }








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

  /*// Store the time, state, and input at knot points
  VectorXd time_at_knots = gm_traj_opt.dircon->GetSampleTimes(result);
  MatrixXd state_at_knots = gm_traj_opt.dircon->GetStateSamples(result);
  MatrixXd input_at_knots = gm_traj_opt.dircon->GetInputSamples(result);
  writeCSV(directory + output_prefix + string("time_at_knots.csv"), time_at_knots);
  writeCSV(directory + output_prefix + string("state_at_knots.csv"), state_at_knots);
  writeCSV(directory + output_prefix + string("input_at_knots.csv"), input_at_knots);
  cout << "time_at_knots = " << time_at_knots << endl;*/


  // visualizer
  /*const PiecewisePolynomial<double> pp_xtraj =
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


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

#include "solvers/optimization_utils.h"
#include "systems/trajectory_optimization/dircon_position_data.h"
#include "systems/trajectory_optimization/dircon_kinematic_data_set.h"
#include "systems/trajectory_optimization/dircon_opt_constraints.h"
#include "systems/trajectory_optimization/hybrid_dircon.h"

#include "multibody/visualization_utils.h"

#include "systems/goldilocks_models/symbolic_manifold.h"
#include "systems/goldilocks_models/file_utils.h"

#include "examples/Goldilocks_models/debug_tools.h"

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

using drake::geometry::SceneGraph;
using drake::multibody::Body;
using drake::multibody::Parser;
using drake::systems::rendering::MultibodyPositionToGeometryPose;


// using Isometry3 = Eigen::Transform<Scalar, 3, Eigen::Isometry>

namespace dairlib {
namespace goldilocks_models {

using systems::trajectory_optimization::HybridDircon;
using systems::trajectory_optimization::DirconDynamicConstraint;
using systems::trajectory_optimization::DirconKinematicConstraint;
using systems::trajectory_optimization::DirconOptions;
using systems::trajectory_optimization::DirconKinConstraintType;
using systems::SubvectorPassThrough;

MathematicalProgramResult trajOptGivenWeights(MultibodyPlant<double> & plant,
    MultibodyPlant<AutoDiffXd> & plant_autoDiff,
    int n_s, int n_sDDot,
    int n_feature_s,
    int n_feature_sDDot,
    Eigen::VectorXd & theta_s, Eigen::VectorXd & theta_sDDot,
    double stride_length, double duration, int max_iter,
    string directory,
    string init_file, std::string prefix,
    vector<VectorXd> & w_sol_vec,
    vector<MatrixXd> & A_vec, vector<MatrixXd> & H_vec,
    vector<VectorXd> & y_vec,
    vector<VectorXd> & lb_vec, vector<VectorXd> & ub_vec,
    vector<VectorXd> & b_vec,
    vector<VectorXd> & c_vec,
    vector<MatrixXd> & B_vec,
    const double & Q_double, const double & R_double,
    double eps_reg,
    bool is_get_nominal) {

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

  // trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
  //                          "Print file", "snopt.out");
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
  /*trajopt->AddQuadraticCost(eps_reg*I_x*timestep/2,VectorXd::Zero(n_x),x0);
  for(int i=1; i<=N-2; i++){
    auto xi = trajopt->state(i);
    trajopt->AddQuadraticCost(eps_reg*I_x*timestep,VectorXd::Zero(n_x),xi);
  }
  trajopt->AddQuadraticCost(eps_reg*I_x*timestep/2,VectorXd::Zero(n_x),xf);*/


  // initial guess if the file exists
  if (!init_file.empty()) {
    MatrixXd w0 = readCSV(directory + init_file);
    trajopt->SetInitialGuessForAllVariables(w0);
  }

  // Move the trajectory optmization problem into GoldilcocksModelTrajOpt
  // where we add the constraints for reduced order model
  GoldilcocksModelTrajOpt gm_traj_opt(
    n_s, n_sDDot, n_feature_s, n_feature_sDDot, theta_s, theta_sDDot,
    std::move(trajopt), &plant_autoDiff, num_time_samples, is_get_nominal);

  // Add regularization term here so that hessian is pd (for outer loop), so
  // that we can use schur complement method
  // TODO(yminchen): should I add to all decision variable? or just state?
  if (!is_get_nominal) {
    auto w = gm_traj_opt.dircon->decision_variables();
    for (int i = 0; i < w.size(); i++)
      gm_traj_opt.dircon->AddQuadraticCost(eps_reg * w(i)*w(i));
  }

  // cout << "Solving DIRCON (based on MultipleShooting)\n";
  auto start = std::chrono::high_resolution_clock::now();
  const MathematicalProgramResult result = Solve(
        *gm_traj_opt.dircon, gm_traj_opt.dircon->initial_guess());
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  cout << "Solve time:" << elapsed.count() << " | ";
  SolutionResult solution_result = result.get_solution_result();
  cout << solution_result <<  " | ";
  cout << "Cost:" << result.get_optimal_cost() << endl;

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
  }*/

  // Testing: see what are the decision varaibles
  // cout << "\nAll decision variable:\n"
  //   << gm_traj_opt.dircon->decision_variables() << endl;






  // Store the time, state, and input at knot points
  VectorXd time_at_knots = gm_traj_opt.dircon->GetSampleTimes(result);
  MatrixXd state_at_knots = gm_traj_opt.dircon->GetStateSamples(result);
  // MatrixXd input_at_knots = gm_traj_opt.dircon->GetInputSamples(result);
  writeCSV(directory + prefix + string("time_at_knots.csv"), time_at_knots);
  writeCSV(directory + prefix + string("state_at_knots.csv"), state_at_knots);
  // writeCSV(directory + prefix + string("input_at_knots.csv"), input_at_knots);






  // Get the solution of all the decision variable
  VectorXd w_sol = result.GetSolution(
                     gm_traj_opt.dircon->decision_variables());
  writeCSV(directory + prefix + string("w.csv"), w_sol);

  // Assume theta is fixed. Get the linear approximation of the cosntraints and
  // second order approximation of the cost.
  if (!is_get_nominal) {
    cout << "\nGetting A, H, y, lb, ub, b.\n";
    MatrixXd A, H;
    VectorXd y, lb, ub, b;
    double c_double;
    solvers::LinearizeConstraints(
      *gm_traj_opt.dircon.get(), w_sol, &y, &A, &lb, &ub);
    c_double = solvers::SecondOrderCost(*gm_traj_opt.dircon.get(), w_sol, &H, &b);
    VectorXd c(1);
    c << c_double;

    // Get matrix B (~get feature vectors)
    cout << "\nGetting B.\n";
    int n_theta_s = theta_s.size();
    int n_theta_sDDot = theta_sDDot.size();
    int n_theta = n_theta_s + n_theta_sDDot;
    MatrixXd B = MatrixXd::Zero(A.rows(), n_theta);
    // Get the row index of B matrix where dynamics constraint starts
    VectorXd ind_head = solvers::GetConstraintRows(
                          *gm_traj_opt.dircon.get(),
                          gm_traj_opt.dynamics_constraint_at_head_bindings[0]);
    // cout << "ind_head = " << ind_head(0) << endl;
    VectorXd ind_tail = solvers::GetConstraintRows(
                          *gm_traj_opt.dircon.get(),
                          gm_traj_opt.dynamics_constraint_at_tail_bindings[0]);
    // cout << "ind_tail = " << ind_tail(0) << endl;
    int N_accum = 0;
    for (unsigned int l = 0; l < num_time_samples.size() ; l++) {
      for (int m = 0; m < num_time_samples[l] - 1 ; m++) {
        int i = N_accum + m;
        cout << "i = " << i << endl;
        // Get the gradient value first
        auto x_i = gm_traj_opt.dircon->state_vars_by_mode(l, m);
        auto x_iplus1 = gm_traj_opt.dircon->state_vars_by_mode(l, m + 1);
        auto h_btwn_knot_i_iplus1 = gm_traj_opt.dircon->timestep(i);
        VectorXd x_i_sol = result.GetSolution(x_i);
        VectorXd x_iplus1_sol = result.GetSolution(x_iplus1);
        VectorXd h_i_sol = result.GetSolution(h_btwn_knot_i_iplus1);

        MatrixXd dyn_gradient_head =
          gm_traj_opt.dynamics_constraint_at_head->getGradientWrtTheta(
            x_i_sol, x_iplus1_sol, h_i_sol);
        MatrixXd dyn_gradient_tail =
          gm_traj_opt.dynamics_constraint_at_tail->getGradientWrtTheta(
            x_i_sol, x_iplus1_sol, h_i_sol);

        // Fill in B matrix
        B.block(ind_head(0) + i * 2 * n_sDDot, 0, n_sDDot, n_theta)
          = dyn_gradient_head;
        B.block(ind_tail(0) + i * 2 * n_sDDot, 0, n_sDDot, n_theta)
          = dyn_gradient_tail;
        // cout << "row " << ind_head(0) + i * 2 * n_sDDot << endl;
        // cout << "row " << ind_tail(0) + i * 2 * n_sDDot << endl << endl;
      }
      N_accum += num_time_samples[l];
      N_accum -= 1;  // due to overlaps between modes
    }

    // Push the solution to the vector
    w_sol_vec.push_back(w_sol);
    H_vec.push_back(H);
    b_vec.push_back(b);
    c_vec.push_back(c);
    A_vec.push_back(A);
    lb_vec.push_back(lb);
    ub_vec.push_back(ub);
    y_vec.push_back(y);
    B_vec.push_back(B);

    // Store the vectors and matrices
    cout << "\nStoring vectors and matrices into csv.\n";
    writeCSV(directory + prefix + string("c.csv"), c);

    /*writeCSV(directory + prefix + string("H.csv"), H);
    writeCSV(directory + prefix + string("b.csv"), b);
    writeCSV(directory + prefix + string("A.csv"), A);
    writeCSV(directory + prefix + string("lb.csv"), lb);
    writeCSV(directory + prefix + string("ub.csv"), ub);
    writeCSV(directory + prefix + string("y.csv"), y);*/
    // writeCSV(directory + prefix + string("B.csv"), B);


    // Store s, ds and dds into csv files
    cout << "\nStoring s, ds and dds into csv.\n";
    std::vector<VectorXd> s_vec;
    std::vector<VectorXd> ds_vec;
    std::vector<VectorXd> dds_vec;
    std::vector<VectorXd> h_vec;
    N_accum = 0;
    // for (unsigned int l = 0; l < num_time_samples.size() ; l++) {
    for (unsigned int l = 0; l < 1 ; l++) { // just look at the first mode now
      for (int m = 0; m < num_time_samples[l]; m++) {
        int i = N_accum + m;
        // Get the gradient value first
        auto x_i = gm_traj_opt.dircon->state_vars_by_mode(l, m);
        VectorXd x_i_sol = result.GetSolution(x_i);

        VectorXd s;
        VectorXd ds;
        gm_traj_opt.dynamics_constraint_at_head->getSAndSDot(x_i_sol, s, ds);
        VectorXd dds = gm_traj_opt.dynamics_constraint_at_head->getSDDot(s, ds);
        s_vec.push_back(s);
        ds_vec.push_back(ds);
        dds_vec.push_back(dds);

        if (m < num_time_samples[l] - 1) {
          auto h_i = gm_traj_opt.dircon->timestep(i);
          VectorXd h_i_sol = result.GetSolution(h_i);
          h_vec.push_back(h_i_sol);
        }
      }
      N_accum += num_time_samples[l];
      N_accum -= 1;  // due to overlaps between modes
    }
    PiecewisePolynomial<double> s_spline = createCubicSplineGivenSAndSdot(
        h_vec, s_vec, ds_vec);
    storeSplineOfS(h_vec, s_spline, directory, prefix);
    // checkSplineOfS(h_vec, dds_vec, s_spline);








    // Below are all for debugging

    // Checking B
    // BTW, the code only work in the case of s = q_1 ^2 and dds = s^3
    bool is_checking_matrix_B = false;
    if (is_checking_matrix_B) {
      N_accum = 0;
      for (unsigned int l = 0; l < 1 ; l++) { // just look at the first mode now
        for (int m = 0; m < num_time_samples[l] - 1; m++) {
          int i = N_accum + m;
          cout << "i = " << i << endl;

          // From finite differencing
          auto x_i = gm_traj_opt.dircon->state_vars_by_mode(l, m);
          auto x_iplus1 = gm_traj_opt.dircon->state_vars_by_mode(l, m + 1);
          auto h_btwn_knot_i_iplus1 = gm_traj_opt.dircon->timestep(i);
          VectorXd x_i_sol = result.GetSolution(x_i);
          VectorXd x_iplus1_sol = result.GetSolution(x_iplus1);
          VectorXd h_i_sol = result.GetSolution(h_btwn_knot_i_iplus1);
          double h_i = h_i_sol(0);

          MatrixXd grad_head_byFD =
            gm_traj_opt.dynamics_constraint_at_head->getGradientWrtTheta(
              x_i_sol, x_iplus1_sol, h_i_sol);
          MatrixXd grad_tail_byFD =
            gm_traj_opt.dynamics_constraint_at_tail->getGradientWrtTheta(
              x_i_sol, x_iplus1_sol, h_i_sol);

          // From hand calculation (theta_s part)
          double phis_i = x_i_sol(1) * x_i_sol(1);
          double dphis_i = 2 * x_i_sol(1) * x_i_sol(1 + 7);
          double phis_iplus1 = x_iplus1_sol(1) * x_iplus1_sol(1);
          double dphis_iplus1 = 2 * x_iplus1_sol(1) * x_iplus1_sol(1 + 7);
          double grad_head_exact =
            (-6 * (phis_i - phis_iplus1) -
             2 * h_i * (2 * dphis_i + dphis_iplus1)) / (h_i * h_i) -
            theta_sDDot(0) * (3 * pow(theta_s(0), 2) * pow(phis_i, 3));
          double grad_tail_exact =
            (6 * (phis_i - phis_iplus1) +
             2 * h_i * (dphis_i + 2 * dphis_iplus1)) / (h_i * h_i) -
            theta_sDDot(0) * (3 * pow(theta_s(0), 2) * pow(phis_iplus1, 3));

          // From hand calculation (theta_sddot part)
          double dyn_feature_i = pow(theta_s(0) * phis_i, 3);
          double dyn_feature_iplus1 = pow(theta_s(0) * phis_iplus1, 3);

          // Compare the values
          cout << grad_head_byFD << " (by finite difference)" << endl;
          cout << grad_head_exact << " " << -dyn_feature_i <<
               " (analytically (exact solution))" << endl;
          cout << "  differnce = " << grad_head_byFD(0, 0) - grad_head_exact <<
               ", " << grad_head_byFD(0, 1) + dyn_feature_i << endl;
          cout << grad_tail_byFD << " (by finite difference)" << endl;
          cout << grad_tail_exact << " " << -dyn_feature_iplus1 <<
               " (analytically (exact solution))" << endl;
          cout << "  differnce = " << grad_tail_byFD(0, 0) - grad_tail_exact <<
               ", " << grad_tail_byFD(0, 1) + dyn_feature_iplus1 << endl;
        }
        N_accum += num_time_samples[l];
        N_accum -= 1;  // due to overlaps between modes
      }
    }

    // Checking the accuracy of s and sdot calculation
    // BTW, the code only work in the case of y = q_1 ^2
    bool is_checking_s_sdot = false;
    if (is_checking_s_sdot) {
      N_accum = 0;
      for (unsigned int l = 0; l < 1 ; l++) { // just look at the first mode now
        for (int m = 0; m < num_time_samples[l] - 1; m++) {
          // int i = N_accum + m;
          // cout << "i = " << i << endl;

          auto x_i = gm_traj_opt.dircon->state_vars_by_mode(l, m);
          auto x_iplus1 = gm_traj_opt.dircon->state_vars_by_mode(l, m + 1);
          VectorXd x_i_sol = result.GetSolution(x_i);
          VectorXd x_iplus1_sol = result.GetSolution(x_iplus1);

          VectorXd s_i;
          VectorXd ds_i;
          VectorXd s_iplus1;
          VectorXd ds_iplus1;
          gm_traj_opt.dynamics_constraint_at_head->getSAndSDot(
            x_i_sol, s_i, ds_i);
          // cout << "ds_i_byhand - ds_i = " <<
          // theta_s(0) * 2 * x_i_sol(1)*x_i_sol(1 + 7) - ds_i(0) << endl;
          gm_traj_opt.dynamics_constraint_at_head->getSAndSDot(
            x_iplus1_sol, s_iplus1, ds_iplus1);
          // cout << "ds_iplus1_byhand - ds_iplus1 = " <<
          // theta_s(0) * 2 * x_iplus1_sol(1)*x_iplus1_sol(1 + 7) - ds_iplus1(0) << endl;
        }
        N_accum += num_time_samples[l];
        N_accum -= 1;  // due to overlaps between modes
      }
    }






    bool is_comparing_two_constraint_linearization = false;
    if (is_comparing_two_constraint_linearization) {
      /*// Comparing the new cosntraint linearization and the old lienarization
      MatrixXd A_new;
      VectorXd y_new, lb_new, ub_new;
      systems::trajectory_optimization::newlinearizeConstraints(
        gm_traj_opt.dircon.get(), w_sol, y_new, A_new, lb_new, ub_new);
      // reorganize the rows or A
      cout << "Reorganize the rows or A\n";
      int nl_i = A.rows();
      int nw_i = A.cols();
      cout << "size of A = " << A.rows() << ", " <<  A.cols() << endl;
      cout << "size of A_new = " << A_new.rows() << ", " <<  A_new.cols() << endl;
      vector<int> new_row_idx;
      VectorXd rowi(nw_i);
      VectorXd rowj(nw_i);
      VectorXd normalized_rowi(nw_i);
      VectorXd normalized_rowj(nw_i);
      for (int i = 0; i < nl_i; i++) { // A_new
        for (int j = 0; j < nl_i; j++) { // A
          rowi = A_new.row(i).transpose();
          rowj = A.row(j).transpose();
          normalized_rowi = rowi / rowi.norm();
          normalized_rowj = rowj / rowj.norm();
          if ((normalized_rowi - normalized_rowj).norm() < 1e-15) {
            if (rowi.norm() != rowj.norm()) {
              cout << i << "-th row of A_new: scale are different by " <<
                   rowi.norm() / rowj.norm() << endl;
            }
            // check ub and lb
            // cout << "("<<i<<","<<j<<")"<<": \n";
            // cout << "  ub(j) = " << ub(j) << endl;
            // cout << "  ub_new(i) = " << ub_new(i) << endl;
            // cout << "  lb(j) = " << lb(j) << endl;
            // cout << "  lb_new(i) = " << lb_new(i) << endl;
            if (ub(j) == ub_new(i) && lb(j) == lb_new(i) ) {
              // Maybe there are duplicated rows, so check if already assigned the row
              bool is_existing = false;
              for (int idx : new_row_idx) {
                if (idx == j) {
                  is_existing = true;
                  break;
                }
              }
              if (!is_existing) {
                new_row_idx.push_back(j);
                break;
              }
            }
          }
          if (j == nl_i - 1) cout << i << "-th row of A_new has no corresponding A\n";
        }
      }
      MatrixXd A_new_reorg(nl_i, nw_i);
      for (int i = 0; i < nl_i; i++) {
        A_new_reorg.row(new_row_idx[i]) = A_new.row(i);
      }
      cout << "size of new_row_idx = " << new_row_idx.size() << endl;
      // Check if new_row_idx covers every index
      for (int i = 0; i < nl_i; i++) {
        for (int j = 0; j < nl_i; j++) {
          if (new_row_idx[j] == i) {
            break;
          }
          if (j == nl_i - 1 ) cout << i << "-th row of A doesn't get assigned to\n";
        }
      }
      // for (int i:new_row_idx) cout << i << endl;

      // compare A with A_new_reorg
      MatrixXd diff_A = A_new_reorg - A;
      MatrixXd abs_diff_A = diff_A.cwiseAbs();
      VectorXd left_one = VectorXd::Ones(abs_diff_A.rows());
      VectorXd right_one = VectorXd::Ones(abs_diff_A.cols());
      cout << "sum-abs-diff_A: " <<
           left_one.transpose()*abs_diff_A*right_one << endl;
      cout << "sum-abs-diff_A divide by m*n: " <<
           left_one.transpose()*abs_diff_A*right_one /
           (abs_diff_A.rows()*abs_diff_A.cols())
           << endl;
      double max_diff_A_element = abs_diff_A(0, 0);
      for (int i = 0; i < abs_diff_A.rows(); i++)
        for (int j = 0; j < abs_diff_A.cols(); j++) {
          if (abs_diff_A(i, j) > max_diff_A_element) {
            max_diff_A_element = abs_diff_A(i, j);
            cout << "(" << i << "," << j << ")" << ": max_diff_A_element = " <<
                 max_diff_A_element << endl;
          }
        }
      cout << "max_diff_A_element = " << max_diff_A_element << endl;*/
    }









    // int A_row = A.rows();
    // int A_col = A.cols();
    // cout << "A_row = " << A_row << endl;
    // cout << "A_col = " << A_col << endl;
    // int max_row_col = (A_row > A_col) ? A_row : A_col;
    // Eigen::BDCSVD<MatrixXd> svd_5(A);
    // cout << "A:\n";
    // cout << "  biggest singular value is " << svd_5.singularValues()(0) << endl;
    // cout << "  smallest singular value is "
    //      << svd_5.singularValues()(max_row_col - 1) << endl;





    /*cout << "\ncheck if H is diagonal: \n";
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
    double tol = 1e-8; //1e-4
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
    int nl_i = A.rows();
    int nw_i = A.cols();
    MathematicalProgram quadprog;
    auto dw = quadprog.NewContinuousVariables(nw_i, "dw");
    quadprog.AddLinearConstraint( A,
                                  lb - y,
                                  ub - y,
                                  dw);
    quadprog.AddQuadraticCost(H, b, dw);
    // quadprog.SetSolverOption(drake::solvers::GurobiSolver::id(), "BarConvTol", 1E-9);
    quadprog.SetSolverOption(drake::solvers::SnoptSolver::id(),
                             "Print file", "snopt.out");
    quadprog.SetSolverOption(drake::solvers::SnoptSolver::id(),
                             "Major iterations limit", 10000);
    // quadprog.SetSolverOption(drake::solvers::SnoptSolver::id(), "Major feasibility tolerance", 1.0e-14); //1.0e-10
    // quadprog.SetSolverOption(drake::solvers::SnoptSolver::id(), "Minor feasibility tolerance", 1.0e-14); //1.0e-10
    const auto result2 = Solve(quadprog);
    auto solution_result2 = result2.get_solution_result();
    cout << solution_result2 << endl;
    cout << "Cost:" << result2.get_optimal_cost() << endl;
    VectorXd dw_sol = result2.GetSolution(quadprog.decision_variables());
    cout << "dw_sol norm:" << dw_sol.norm() << endl;
    // cout << "dw_sol = \n" << dw_sol << endl;
    cout << "This should be zero\n" << VectorXd::Ones(nl_i).transpose()*A*dw_sol <<
         endl; // dw in null space
    cout << "if this is not zero, then w=0 is not optimal: " << dw_sol.transpose()*b
         << endl;
    cout << "Finished traj opt\n\n";

    // vector<double> w_sol_sort;
    // for(int i=0; i<dw_sol.size(); i++){
    //   w_sol_sort.push_back(dw_sol(i));
    // }
    // std::sort(w_sol_sort.begin(), w_sol_sort.end());
    // for(double w_sol_sort_ele : w_sol_sort)
    //   cout << w_sol_sort_ele << endl;

    // // Check if dw=0 violates any constraint
    // VectorXd QP_ub = ub-y;
    // VectorXd QP_lb = lb-y;
    // for(int i=0; i<nl_i; i++){
    //   if(QP_ub(i)<0)
    //     cout<< "row " << i << ": upper bound is smaller than 0 by " << QP_ub(i) << endl;
    //   if(QP_lb(i)>0)
    //     cout<< "row " << i << ": lower bound is larger than 0 by " << QP_lb(i) << endl;
    // }
    // cout << endl;
    // // Check if dw* to the QP violates any constraint
    // VectorXd QP_constraint_val = A*dw_sol;
    // for(int i=0; i<nl_i; i++){
    //   if(QP_ub(i) < QP_constraint_val(i))
    //     cout<< "row " << i << ": upper bound constraint violation by " << QP_constraint_val(i) - QP_ub(i) << endl;
    //   if(QP_lb(i) > QP_constraint_val(i))
    //     cout<< "row " << i << ": lower bound constraint violation by " << QP_constraint_val(i) - QP_lb(i) << endl;
    // }
    // cout << endl;




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
      c_nonlinear = solvers::SecondOrderCost(
                      *gm_traj_opt.dircon.get(), w_sol_test, &H2, &b2) - c_double;
      cout << "i = " << i << endl;
      cout << "  c_nonlinear = " << c_nonlinear << endl;
      VectorXd dw_sol_test = i * eps * dw_sol;
      double c_aquadprog = 0.5 * dw_sol_test.transpose() * H * dw_sol_test + b.dot(
                             dw_sol_test) + c_double - c_double;
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
        solvers::LinearizeConstraints(
          *gm_traj_opt.dircon.get(), w_sol_test, &y2, &A2, &lb2, &ub2);
        unsigned int k = 0;
        for (unsigned int j = 0; j < n_ae; j++) {
          double violation = y2(active_eq_row_idx[j]) - ub(active_eq_row_idx[j]);
          if (abs(violation) > 1e-8) {
            constraint_vio_row_idx.push_back(j);
            k++;
            if (k == n_show)
              break;
          }
          if (i == n_show - 1 && j == n_ae - 1 && k < n_show) {
            cout << "There are only " << k << " # of violations\n";
          }
        }
        if (constraint_vio_row_idx.size() >= n_show)
          break;
        else if (i != 10 - 1)
          constraint_vio_row_idx.clear();
      }
      cout << "  Row index of violation = ";
      for (int j : constraint_vio_row_idx) {
        cout << j << ", ";
      }
      cout << endl;
      // evaluate the chosen rows
      for (unsigned int i = 0; i < 10 ; i++) {
        cout << "i = " << i << endl;
        VectorXd w_sol_test = w_sol + i * eps * dw_sol;
        MatrixXd A2;
        VectorXd y2, lb2, ub2;
        solvers::LinearizeConstraints(
          *gm_traj_opt.dircon.get(), w_sol_test, &y2, &A2, &lb2, &ub2);
        cout << "  nonlinear_constraint_val = ";
        for (int j : constraint_vio_row_idx) {
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
        solvers::LinearizeConstraints(
          *gm_traj_opt.dircon.get(), w_sol_test, &y2, &A2, &lb2, &ub2);
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
        solvers::LinearizeConstraints(
          *gm_traj_opt.dircon.get(), w_sol_test, &y2, &A2, &lb2, &ub2);
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
    }*/
  }  // end of if(!is_get_nominal)






  return result;
}

}  // namespace goldilocks_models
}  // namespace dairlib


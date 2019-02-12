#include "examples/Goldilocks_models/traj_opt_given_weigths.h"

#include <gflags/gflags.h>

#include <memory>
#include <chrono>

#include "drake/multibody/rigid_body_tree_construction.h"
#include "drake/multibody/joints/floating_base_types.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_tree.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"

#include "drake/multibody/rigid_body_plant/drake_visualizer.h"
#include "drake/lcm/drake_lcm.h"

#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/constraint.h"
#include "systems/trajectory_optimization/dircon_util.h"
#include "systems/trajectory_optimization/dircon_position_data.h"
#include "systems/trajectory_optimization/dircon_kinematic_data_set.h"
#include "systems/trajectory_optimization/dircon_opt_constraints.h"

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
using std::vector;
using std::shared_ptr;
using std::cout;
using std::endl;
using std::string;

namespace dairlib {
namespace goldilocks_models {

using systems::trajectory_optimization::HybridDircon;
using systems::trajectory_optimization::DirconDynamicConstraint;
using systems::trajectory_optimization::DirconKinematicConstraint;
using systems::trajectory_optimization::DirconOptions;
using systems::trajectory_optimization::DirconKinConstraintType;

void trajOptGivenWeights(double stride_length, double duration, int iter, 
                         string directory,
                         string init_file, 
                         string weights_file,
                         string output_prefix) {

  RigidBodyTree<double> tree;
  drake::parsers::urdf::AddModelInstanceFromUrdfFileToWorld(
      "examples/PlanarWalker/PlanarWalkerWithTorso.urdf", drake::multibody::joints::kFixed, &tree);

// world
// base
// base_x
// base_xz
// torso
// torso_mass
// left_upper_leg
// left_upper_leg_mass
// left_lower_leg
// left_lower_leg_mass
// right_upper_leg
// right_upper_leg_mass
// right_lower_leg
// right_lower_leg_mass

// left_hip_torque
// right_hip_torque
// left_knee_torque
// right_knee_torque

// planar_x
// planar_z
// planar_roty
// left_hip_pin
// left_knee_pin
// right_hip_pin
// right_knee_pin

// planar_xdot
// planar_zdot
// planar_rotydot
// left_hip_pindot
// left_knee_pindot
// right_hip_pindot
// right_knee_pindot


  int n_q = tree.get_num_positions();
  int n_v = tree.get_num_velocities();
  int n_x = n_q + n_v;
  int n_u = tree.get_num_actuators();
  std::cout<<"n_x = "<<n_x<<"\n";
  std::cout<<"n_u = "<<n_u<<"\n";

  int leftLegIdx = tree.FindBodyIndex("left_lower_leg");
  int rightLegIdx = tree.FindBodyIndex("right_lower_leg");

  Vector3d pt;
  pt << 0, 0, -.5;
  bool isXZ = true;

  auto leftFootConstraint = DirconPositionData<double>(tree, leftLegIdx, pt,
                                                       isXZ);
  auto rightFootConstraint = DirconPositionData<double>(tree, rightLegIdx, pt,
                                                        isXZ);

  Vector3d normal;
  normal << 0, 0, 1;
  double mu = 1;
  leftFootConstraint.addFixedNormalFrictionConstraints(normal, mu);
  rightFootConstraint.addFixedNormalFrictionConstraints(normal, mu);
    // std::cout<<leftFootConstraint.getLength()<<"\n"; //2 dim. I guess the contact point constraint in the x and z direction

  std::vector<DirconKinematicData<double>*> leftConstraints;
  leftConstraints.push_back(&leftFootConstraint);
  auto leftDataSet = DirconKinematicDataSet<double>(tree, &leftConstraints);

  std::vector<DirconKinematicData<double>*> rightConstraints;
  rightConstraints.push_back(&rightFootConstraint);
  auto rightDataSet = DirconKinematicDataSet<double>(tree, &rightConstraints);

  auto leftOptions = DirconOptions(leftDataSet.countConstraints());
  leftOptions.setConstraintRelative(0, true); //TODO: ask what is relative constraint here?
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
  min_dt.push_back(.01);   // bound for time difference between adjacent samples in the first mode // See HybridDircon constructor
  min_dt.push_back(.01);   // bound for time difference between adjacent samples in the second mode
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

  auto trajopt = std::make_shared<HybridDircon<double>>(tree, num_time_samples, min_dt,
                                                        max_dt, dataset_list,
                                                        options_list);

  // trajopt->AddDurationBounds(duration, duration); // You can comment this out to not put any constraint on the time

  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Print file", "snopt.out");
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(),
                           "Major iterations limit", iter);
  trajopt->SetSolverOption(drake::solvers::SnoptSolver::id(), "Verify level",
                           0);

  // Periodicity constraints
  // planar_x - 0
  // planar_z - 1
  // planar_rot - 2
  // left_hip_pin - 3
  // left_knee_pin - 4
  // right_hip_pin - 5
  // right_knee_pin - 6
  auto x0 = trajopt->initial_state();
  // auto xf = trajopt->final_state();
  auto xf = trajopt->state_vars_by_mode(num_time_samples.size()-1,
                                        num_time_samples[num_time_samples.size()-1]-1);

  trajopt->AddLinearConstraint(x0(1) == xf(1));
  trajopt->AddLinearConstraint(x0(2) == xf(2));
  trajopt->AddLinearConstraint(x0(3) == xf(5));
  trajopt->AddLinearConstraint(x0(4) == xf(6));
  trajopt->AddLinearConstraint(x0(5) == xf(3));
  trajopt->AddLinearConstraint(x0(6) == xf(4));

  trajopt->AddLinearConstraint(x0(7) == xf(7));
  trajopt->AddLinearConstraint(x0(8) == xf(8));
  trajopt->AddLinearConstraint(x0(9) == xf(9));
  trajopt->AddLinearConstraint(x0(10) == xf(12));
  trajopt->AddLinearConstraint(x0(11) == xf(13));
  trajopt->AddLinearConstraint(x0(12) == xf(10));
  trajopt->AddLinearConstraint(x0(13) == xf(11));

  // u periodic constraint
  auto u0 = trajopt->input(0);
  auto uf = trajopt->input(N-1);
  trajopt->AddLinearConstraint(u0(0) == uf(1));
  trajopt->AddLinearConstraint(u0(1) == uf(0));
  trajopt->AddLinearConstraint(u0(2) == uf(3));
  trajopt->AddLinearConstraint(u0(3) == uf(2));

  // Knee joint limits
  auto x = trajopt->state();
  trajopt->AddConstraintToAllKnotPoints(x(4) >= 5.0/180.0*M_PI);
  trajopt->AddConstraintToAllKnotPoints(x(6) >= 5.0/180.0*M_PI);
  trajopt->AddConstraintToAllKnotPoints(x(4) <= M_PI/2.0);
  trajopt->AddConstraintToAllKnotPoints(x(6) <= M_PI/2.0);

  // hip joint limits
  trajopt->AddConstraintToAllKnotPoints(x(3) >= -M_PI/2);
  trajopt->AddConstraintToAllKnotPoints(x(5) >= -M_PI/2);
  trajopt->AddConstraintToAllKnotPoints(x(3) <= M_PI/2.0);
  trajopt->AddConstraintToAllKnotPoints(x(5) <= M_PI/2.0);


  // x-distance constraint constraints
  trajopt->AddLinearConstraint(x0(0) == 0);
  trajopt->AddLinearConstraint(xf(0) == stride_length);

  const double R = 10;  // Cost on input effort
  auto u = trajopt->input();
  trajopt->AddRunningCost(u.transpose()*R*u);
  MatrixXd Q = MatrixXd::Zero(2*n_q, 2*n_q);
  for (int i=0; i < n_q; i++) {
    Q(i+n_q, i+n_q) = 10;
  }
  trajopt->AddRunningCost(x.transpose()*Q*x);

  // initial guess if the file exists
  if (!init_file.empty()) {
    MatrixXd z0 = readCSV(directory + init_file);
    trajopt->SetInitialGuessForAllVariables(z0);
  }
  // if(true){
    // // TODO: maybe should construct the pp from coefficient

  //   VectorXd time_at_knot_point = readCSV(directory + "init_traj_time_at_knots.csv");
  //   MatrixXd state_at_knot_point = readCSV(directory + "init_traj_state_at_knots.csv");
  //   MatrixXd input_at_knot_point = readCSV(directory + "init_traj_input_at_knots.csv");
  //     // std::cout<<time_at_knot_point.rows()<<", "<<time_at_knot_point.cols()<<"\n";
  //     // std::cout<<"time_at_knot_point = \n"<<time_at_knot_point<<"\n";
  //     // std::cout<<state_at_knot_point.rows()<<", "<<state_at_knot_point.cols()<<"\n";
  //     // std::cout<<"state_at_knot_point = \n"<<state_at_knot_point<<"\n";
  //     // std::cout<<input_at_knot_point.rows()<<", "<<input_at_knot_point.cols()<<"\n";
  //     // std::cout<<"input_at_knot_point = \n"<<input_at_knot_point<<"\n";
  //   int n_init_knots = time_at_knot_point.size();

  //   std::vector<double> T_knotpoint(n_init_knots, 0);
  //   std::vector<MatrixXd> Y_state(n_init_knots, MatrixXd::Zero(state_at_knot_point.rows(), 1));
  //   std::vector<MatrixXd> Y_input(n_init_knots, MatrixXd::Zero(input_at_knot_point.rows(), 1));
  //   for(int i=0;i<n_init_knots;i++){
  //     T_knotpoint[i] = time_at_knot_point(i);
  //     Y_state[i] = state_at_knot_point.col(i);
  //     Y_input[i] = input_at_knot_point.col(i);
  //     // std::cout<<"T_knotpoint = "<<T_knotpoint[i]<<"\n";
  //     // std::cout<<"Y_input = "<<Y_input[i]<<"\n";
  //   }

  //   PiecewisePolynomial<double> traj_init_x = PiecewisePolynomial<double>::Pchip(T_knotpoint, Y_state);
  //   PiecewisePolynomial<double> traj_init_u = PiecewisePolynomial<double>::Pchip(T_knotpoint, Y_input);

  //   trajopt->SetInitialTrajectory(traj_init_u, traj_init_x);
  // }

  std::cout<<"Solving DIRCON (based on MultipleShooting)\n";
  auto start = std::chrono::high_resolution_clock::now();
  auto result = trajopt->Solve();
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  // trajopt->PrintSolution();
  std::cout << "Solve time:" << elapsed.count() <<std::endl;
  std::cout << result << std::endl;
  std::cout << "Cost:" << trajopt->GetOptimalCost() <<std::endl;

  // store the solution of the decision variable 
  VectorXd z = trajopt->GetSolution(trajopt->decision_variables()); //solution of all decision variables 
  writeCSV(directory + output_prefix + string("z.csv"), z);

  // store the time, state, and input at knot points
  // VectorXd time_at_knot_point = trajopt->GetSampleTimes();
  // MatrixXd state_at_knot_point = trajopt->GetStateSamples();
  // MatrixXd input_at_knot_point = trajopt->GetInputSamples();
  // writeCSV(directory + string("init_traj_time_at_knots.csv"), time_at_knot_point);
  // writeCSV(directory + string("init_traj_state_at_knots.csv"), state_at_knot_point);
  // writeCSV(directory + string("init_traj_input_at_knots.csv"), input_at_knot_point);
    // std::cout<<"time_at_knot_point = \n"<<time_at_knot_point<<"\n";
    // std::cout<<state_at_knot_point.rows()<<", "<<state_at_knot_point.cols()<<"\n";
    // std::cout<<"state_at_knot_point = \n"<<state_at_knot_point<<"\n";
    // std::cout<<input_at_knot_point.rows()<<", "<<input_at_knot_point.cols()<<"\n";
    // std::cout<<"input_at_knot_point = \n"<<input_at_knot_point<<"\n";


  // visualizer
  drake::lcm::DrakeLcm lcm;
  drake::systems::DiagramBuilder<double> builder;
  const PiecewisePolynomial<double> pp_xtraj = trajopt->ReconstructStateTrajectory();
  auto state_source = builder.AddSystem<drake::systems::TrajectorySource>(pp_xtraj);
  auto publisher = builder.AddSystem<drake::systems::DrakeVisualizer>(tree, &lcm);
  publisher->set_publish_period(1.0 / 60.0);
  builder.Connect(state_source->get_output_port(),
                  publisher->get_input_port(0));

  auto diagram = builder.Build();


  while (true) {
    drake::systems::Simulator<double> simulator(*diagram);
    simulator.set_target_realtime_rate(.1);
    simulator.Initialize();
    simulator.StepTo(pp_xtraj.end_time());
  }

  return ;
}

}  // namespace goldilocks_models
}  // namespace dairlib


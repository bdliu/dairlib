#include "examples/Goldilocks_models/goldilocks_model_traj_opt.h"


namespace dairlib {
namespace goldilocks_models {

namespace {
VectorXDecisionVariable MakeNamedVariables(const std::string& prefix,
    int num) {
  VectorXDecisionVariable vars(num);
  for (int i = 0; i < num; i++)
    vars(i) = Variable(prefix + std::to_string(i));
  return vars;
}
}  // end of unnamed namespace

// Constructor
GoldilcocksModelTrajOpt::GoldilcocksModelTrajOpt(
  std::unique_ptr<HybridDircon<double>> Dircon_traj_opt_in,
  const MultibodyPlant<double>& plant,
  int N) {

  Dircon_traj_opt = std::move(Dircon_traj_opt_in);
  num_knots_ = N;

  // parameters
  int n_z = 4;
  int n_feature = 5;
  int n_theta = n_z * n_feature;

  // Create model parameter theta as decision variable
  theta_vars_ = Dircon_traj_opt->NewContinuousVariables(n_theta, "theta");
  // Create state z as decision variable
  z_vars_ = Dircon_traj_opt->NewContinuousVariables(n_z * N, "z");
  // placeholder_z_vars_ = MakeNamedVariables("z", n_z);

  // Create kinematics constraint (pointer)
  auto kinematics_constraint = make_shared<KinematicsConstraint>(
                                 n_z, n_feature, n_theta, plant);

  // Add kinematics constraint for all knots
  for (int i = 0; i < N ; i++) {
    auto z_at_knot_i = reduced_model_state(i, n_z);
    auto x_at_knot_i = Dircon_traj_opt->state(i);
    Dircon_traj_opt->AddConstraint(kinematics_constraint,
    {z_at_knot_i, theta_vars_, x_at_knot_i});
  }

}  // end of constructor


Eigen::VectorBlock<const VectorXDecisionVariable>
GoldilcocksModelTrajOpt::reduced_model_state(int index, int n_z) const {
  DRAKE_DEMAND(index >= 0 && index < num_knots_);
  return z_vars_.segment(index * n_z, n_z);
}

void GoldilcocksModelTrajOpt::solve() {};

// https://github.com/RobotLocomotion/drake/blob/master/systems/trajectory_optimization/multiple_shooting.cc
// https://drake.mit.edu/doxygen_cxx/classdrake_1_1systems_1_1trajectory__optimization_1_1_multiple_shooting.html#a3d57e7972ccf310e19d3b73cac1c2a8c


// Inside construct:(
// pass in the multipleShooting class
// pass in the number of knots
// pass in z constraint class
// pass in zdot constraint class
//)

// add new decision variable z
// In for loop
// add constraint for z
// (decision variables passed into the constraint is {x_i,z_i})
// In for loop
// You will need to do direct collocation for zDot (cubic spline)
// 1. Get the spline from z0,z1,
//    zDot0(functino of z0),zDot1(function of z1)
// 2. The constraint is that at the middle point, the slope still match


// methods
// public:
// solve()
// solve MultipleShooting and then return the solution
// private:
// functions related to placeholder_z_vars_


// members:
// public:
// DIRCON_traj_opt
// privite:
// z_vars_
// placeholder_z_vars_







}  // namespace goldilocks_models
}  // namespace dairlib


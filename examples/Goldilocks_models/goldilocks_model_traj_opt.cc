#include "examples/Goldilocks_models/goldilocks_model_traj_opt.h"


namespace dairlib {
namespace goldilocks_models {

// Constructor
GoldilcocksModelTrajOpt::GoldilcocksModelTrajOpt(
  std::unique_ptr<HybridDircon<double>> Dircon_traj_opt_in,
  const MultibodyPlant<double>& plant,
  const std::vector<int> & num_time_samples) {

  // Get total sample ponits
  int N = 0;
  for (uint i = 0; i < num_time_samples.size(); i++)
    N += num_time_samples[i];
  N -= num_time_samples.size() - 1; //Overlaps between modes

  // Members assignment
  Dircon_traj_opt = std::move(Dircon_traj_opt_in);
  num_knots_ = N;

  // parameters
  int n_z = 4;
  int n_zDot = n_z; // Assume that are the same (no quaternion)
  int n_featureZ = 1; // This should match with the dimension of the feature,
                      // since we are hard coding it now. (same below)
  int n_featureZDot = 1;
  int n_thetaZ = n_z * n_featureZ;
  int n_thetaZDot = (n_zDot/2) * n_featureZDot;
      // Assuming position and velocity has the same dimension
      // for the reduced order model.

  // Create decision variables
  z_vars_ = Dircon_traj_opt->NewContinuousVariables(n_z * N, "z");
  thetaZ_vars_ = Dircon_traj_opt->NewContinuousVariables(n_thetaZ, "thetaZ");
  thetaZDot_vars_ = Dircon_traj_opt->NewContinuousVariables(
      n_thetaZDot, "thetaZDot");

  // Create kinematics/dynamics constraint (pointer)
  auto kinematics_constraint = make_shared<KinematicsConstraint>(
                                 n_z, n_featureZ, n_thetaZ, plant);
  auto dynamics_constraint = make_shared<DynamicsConstraint>(
                                 n_zDot, n_featureZDot, n_thetaZDot, plant);

  // Add kinematics constraint for all knots
  // TODO(yminchen): check if kinematics constraint is implemented correctly
  for (int i = 0; i < N ; i++) {
    auto z_at_knot_i = reduced_model_state(i, n_z);
    auto x_at_knot_i = Dircon_traj_opt->state(i);
    Dircon_traj_opt->AddConstraint(kinematics_constraint,
    {z_at_knot_i, thetaZ_vars_, x_at_knot_i});
  }

  // Add dynamics constraint for all segments (between knots) except the last
  // segment of each mode
  // TODO(yminchen): check if dynamics constraint is implemented correctly
  int N_accum = 0;
  for (int i = 0; i < num_time_samples.size() ; i++) {
    // cout << "i = " << i << endl;
    // cout << "N_accum = " << N_accum << endl;
    for (int j = 0; j < num_time_samples[i]-2 ; j++) {
        // -2 because we do not add constraint for the last segment because of
        // discrete dynamics involved
        // TODO(yminchen): can I fix this?
      // cout << "    j = " << j << endl;

      auto z_at_knot_i = reduced_model_state(N_accum+j, n_z);
      auto z_at_knot_iplus1 = reduced_model_state(N_accum+j+1, n_z);
      auto h_btwn_knot_i_iplus1 = Dircon_traj_opt->timestep(N_accum+j);
      Dircon_traj_opt->AddConstraint(dynamics_constraint,
      {z_at_knot_i, z_at_knot_iplus1, thetaZDot_vars_, h_btwn_knot_i_iplus1});
    }

    N_accum += num_time_samples[i];
    N_accum -= 1;  // due to overlaps between modes
  }
}  // end of constructor


Eigen::VectorBlock<const VectorXDecisionVariable>
GoldilcocksModelTrajOpt::reduced_model_state(int index, int n_z) const {
  DRAKE_DEMAND(index >= 0 && index < num_knots_);
  return z_vars_.segment(index * n_z, n_z);
}


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



}  // namespace goldilocks_models
}  // namespace dairlib


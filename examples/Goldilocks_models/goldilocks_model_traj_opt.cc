#include "examples/Goldilocks_models/goldilocks_model_traj_opt.h"


namespace dairlib {
namespace goldilocks_models {

// Constructor
GoldilcocksModelTrajOpt::GoldilcocksModelTrajOpt(
  int n_s, int n_sDDot, int n_feature_s, int n_feature_sDDot,
  VectorXd & theta_s, VectorXd & theta_sDDot,
  std::unique_ptr<HybridDircon<double>> dircon_in,
  const MultibodyPlant<AutoDiffXd> * plant,
  const std::vector<int> & num_time_samples):
  n_s_(n_s),
  n_sDDot_(n_sDDot),
  n_feature_s_(n_feature_s),
  n_feature_sDDot_(n_feature_sDDot) {

  // Get total sample ponits
  int N = 0;
  for (uint i = 0; i < num_time_samples.size(); i++)
    N += num_time_samples[i];
  N -= num_time_samples.size() - 1; //Overlaps between modes

  // Members assignment
  dircon = std::move(dircon_in);
  num_knots_ = N;

  // Create decision variables
  z_vars_ = dircon->NewContinuousVariables(n_s * N, "z");

  // Create kinematics/dynamics constraint (pointer)
  kinematics_constraint = make_shared<KinematicsConstraint>(
                                 n_s, n_feature_s, theta_s, plant);
  dynamics_constraint = make_shared<DynamicsConstraint>(
                                 n_sDDot, n_feature_sDDot, theta_sDDot, plant);

  // Add kinematics constraint for all knots
  // TODO(yminchen): check if kinematics constraint is implemented correctly
  for (int i = 0; i < N ; i++) {
    auto z_at_knot_i = reduced_model_state(i, n_s);
    auto x_at_knot_i = dircon->state(i);
    kinematics_constraint_bindings.push_back(dircon->AddConstraint(
      kinematics_constraint,{z_at_knot_i, x_at_knot_i}));
  }

  // Add dynamics constraint for all segments (between knots) except the last
  // segment of each mode
  // // Dynamics constraint waw tested with fix height acceleration.
  // // Set z = [y;dy] and set dz = [dy;0];
  int N_accum = 0;
  for (unsigned int i = 0; i < num_time_samples.size() ; i++) {
    // cout << "i = " << i << endl;
    // cout << "N_accum = " << N_accum << endl;
    for (int j = 0; j < num_time_samples[i]-2 ; j++) {
        // -2 because we do not add constraint for the last segment because of
        // discrete dynamics involved
        // TODO(yminchen): can I fix this?

      // cout << "    j = " << j << endl;

      auto z_at_knot_k = reduced_model_state(N_accum+j, n_s);
      auto z_at_knot_kplus1 = reduced_model_state(N_accum+j+1, n_s);
      auto h_btwn_knot_k_iplus1 = dircon->timestep(N_accum+j);
      dynamics_constraint_bindings.push_back(dircon->AddConstraint(
        dynamics_constraint, {z_at_knot_k, z_at_knot_kplus1,
          h_btwn_knot_k_iplus1}));
    }

    N_accum += num_time_samples[i];
    N_accum -= 1;  // due to overlaps between modes
  }


}  // end of constructor


Eigen::VectorBlock<const VectorXDecisionVariable>
GoldilcocksModelTrajOpt::reduced_model_state(int index, int n_s) const {
  DRAKE_DEMAND(index >= 0 && index < num_knots_);
  return z_vars_.segment(index * n_s, n_s);
}


}  // namespace goldilocks_models
}  // namespace dairlib


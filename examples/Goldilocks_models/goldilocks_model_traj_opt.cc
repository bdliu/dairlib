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
  // s_vars_ = dircon->NewContinuousVariables(n_s * N, "s");

  // Create kinematics/dynamics constraint (pointer)
  // kinematics_constraint = make_shared<KinematicsConstraint>(
  //                                n_s, n_feature_s, theta_s, plant);
  dynamics_constraint_at_head = make_shared<DynamicsConstraint>(
                                 n_s, n_feature_s, theta_s,
                                 n_sDDot, n_feature_sDDot, theta_sDDot, plant, true);
  dynamics_constraint_at_tail = make_shared<DynamicsConstraint>(
                                 n_s, n_feature_s, theta_s,
                                 n_sDDot, n_feature_sDDot, theta_sDDot, plant, false);

  /*// Add kinematics constraint for all knots
  // TODO(yminchen): check if kinematics constraint is implemented correctly
  int n_q = plant->num_positions();
  for (int i = 0; i < N ; i++) {
    auto s_at_knot_i = reduced_model_position(i, n_s);
    auto x_at_knot_i = dircon->state(i);
    kinematics_constraint_bindings.push_back(dircon->AddConstraint(
      kinematics_constraint,{s_at_knot_i, x_at_knot_i.head(n_q)}));
  }*/

  // Add dynamics constraint for all segments (between knots)
  // // Dynamics constraint waw tested with fix height acceleration.
  // // Set s = [y;dy] and set dz = [dy;0];
  int N_accum = 0;
  for (unsigned int i = 0; i < num_time_samples.size() ; i++) {
    // cout << "i = " << i << endl;
    // cout << "N_accum = " << N_accum << endl;
    for (int j = 0; j < num_time_samples[i]-1 ; j++) {
      // cout << "    j = " << j << endl;
      auto x_at_knot_k = dircon->state(N_accum+j);
      auto x_at_knot_kplus1 = dircon->state(N_accum+j+1);
      auto h_btwn_knot_k_iplus1 = dircon->timestep(N_accum+j);
      dynamics_constraint_at_head_bindings.push_back(dircon->AddConstraint(
        dynamics_constraint_at_head, {x_at_knot_k, x_at_knot_kplus1,
          h_btwn_knot_k_iplus1}));
    }

    N_accum += num_time_samples[i];
    N_accum -= 1;  // due to overlaps between modes
  }

  // Add dynamics constraint for all segments (between knots)
  // // Dynamics constraint waw tested with fix height acceleration.
  // // Set s = [y;dy] and set dz = [dy;0];
  N_accum = 0;
  for (unsigned int i = 0; i < num_time_samples.size() ; i++) {
    // cout << "i = " << i << endl;
    // cout << "N_accum = " << N_accum << endl;
    for (int j = 0; j < num_time_samples[i]-1 ; j++) {
      // cout << "    j = " << j << endl;
      auto x_at_knot_k = dircon->state(N_accum+j);
      auto x_at_knot_kplus1 = dircon->state(N_accum+j+1);
      auto h_btwn_knot_k_iplus1 = dircon->timestep(N_accum+j);
      dynamics_constraint_at_tail_bindings.push_back(dircon->AddConstraint(
        dynamics_constraint_at_tail, {x_at_knot_k, x_at_knot_kplus1,
          h_btwn_knot_k_iplus1}));
    }

    N_accum += num_time_samples[i];
    N_accum -= 1;  // due to overlaps between modes
  }

}  // end of constructor


// Eigen::VectorBlock<const VectorXDecisionVariable>
// GoldilcocksModelTrajOpt::reduced_model_position(int index, int n_s) const {
//   DRAKE_DEMAND(index >= 0 && index < num_knots_);
//   return s_vars_.segment(index * n_s, n_s);
// }


}  // namespace goldilocks_models
}  // namespace dairlib


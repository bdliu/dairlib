#include "examples/Goldilocks_models/dummy_constraint.h"


namespace dairlib {


DummyConstraint::DummyConstraint(const
                                 MultibodyPlant<double>& plant,
                                 const std::string& description):
  Constraint(4,
             16,  // plant.get_num_positions(),
             VectorXd::Zero(4),
             VectorXd::Zero(4),
             description),
  plant_(plant) {
}


void DummyConstraint::DoEval(const
                             Eigen::Ref<const Eigen::VectorXd>& q,
                             Eigen::VectorXd* y) const {
  AutoDiffVecXd y_t;
  Eval(initializeAutoDiff(q), &y_t);
  *y = autoDiffToValueMatrix(y_t);
}

void DummyConstraint::DoEval(const
                             Eigen::Ref<const AutoDiffVecXd>& q,
                             AutoDiffVecXd* y) const {

  *y = initializeAutoDiff(VectorXd::Zero(4));

  // const AutoDiffVecXd q_autodiff = q.head(plant_.get_num_positions());
  // const VectorXd q_double = DiscardGradient(q);

  // KinematicsCache<AutoDiffXd> k_cache = plant_.doKinematics(q_autodiff);
  // KinematicsCache<double> k_cache_double = plant_.doKinematics(q_double);

  // // Collision
  // VectorXd phi_total;
  // Matrix3Xd normal_total, xA_total, xB_total;
  // vector<int> idxA_total, idxB_total;

  // // This (const cast) is an ugly way of doing it. Change it later if a better method is available
  // const_cast<MultibodyPlant<double>&>(plant_).collisionDetect(
  //   k_cache_double, phi_total, normal_total, xA_total, xB_total, idxA_total,
  //   idxB_total);

  // const int num_total_contacts = normal_total.cols();
  // // 4 contacts for Cassie (2 in each toe)
  // const int num_contacts = 4;

  // //Getting the indices of the world and toes
  // const int world_ind = GetBodyIndexFromName(plant_, "world");
  // const int toe_left_ind = GetBodyIndexFromName(plant_, "toe_left");
  // const int toe_right_ind = GetBodyIndexFromName(plant_, "toe_right");

  // vector<int> contact_ind(num_contacts);
  // int k = 0;
  // for (int i = 0; i < num_total_contacts; i++) {
  //   int ind_a = idxA_total.at(i);
  //   int ind_b = idxB_total.at(i);
  //   if ((ind_a == world_ind && ind_b == toe_left_ind) ||
  //       (ind_a == world_ind && ind_b == toe_right_ind) ||
  //       (ind_a == toe_left_ind && ind_b == world_ind) ||
  //       (ind_a == toe_right_ind && ind_b == world_ind)) {

  //     contact_ind.at(k) = i;
  //     k++;

  //   }
  // }

  // auto y_tmp = initializeAutoDiff(VectorXd::Zero(4));

  // // Contact points on body A and B

  // auto contact_A_pt_1 = plant_.transformPoints(k_cache,
  //                       xA_total.col(contact_ind.at(0)),
  //                       idxA_total.at(contact_ind.at(0)),
  //                       world_ind);
  // auto contact_A_pt_2 = plant_.transformPoints(k_cache,
  //                       xA_total.col(contact_ind.at(1)),
  //                       idxA_total.at(contact_ind.at(1)),
  //                       world_ind);
  // auto contact_A_pt_3 = plant_.transformPoints(k_cache,
  //                       xA_total.col(contact_ind.at(2)),
  //                       idxA_total.at(contact_ind.at(2)),
  //                       world_ind);
  // auto contact_A_pt_4 = plant_.transformPoints(k_cache,
  //                       xA_total.col(contact_ind.at(3)),
  //                       idxA_total.at(contact_ind.at(3)),
  //                       world_ind);

  // auto contact_B_pt_1 = plant_.transformPoints(k_cache,
  //                       xB_total.col(contact_ind.at(0)),
  //                       idxB_total.at(contact_ind.at(0)),
  //                       world_ind);
  // auto contact_B_pt_2 = plant_.transformPoints(k_cache,
  //                       xB_total.col(contact_ind.at(1)),
  //                       idxB_total.at(contact_ind.at(1)),
  //                       world_ind);
  // auto contact_B_pt_3 = plant_.transformPoints(k_cache,
  //                       xB_total.col(contact_ind.at(2)),
  //                       idxB_total.at(contact_ind.at(2)),
  //                       world_ind);
  // auto contact_B_pt_4 = plant_.transformPoints(k_cache,
  //                       xB_total.col(contact_ind.at(3)),
  //                       idxB_total.at(contact_ind.at(3)),
  //                       world_ind);

  // // Computing distance

  // y_tmp(0) = (contact_A_pt_1 - contact_B_pt_1).dot(contact_A_pt_1 -
  //            contact_B_pt_1);
  // y_tmp(1) = (contact_A_pt_2 - contact_B_pt_2).dot(contact_A_pt_2 -
  //            contact_B_pt_2);
  // y_tmp(2) = (contact_A_pt_3 - contact_B_pt_3).dot(contact_A_pt_3 -
  //            contact_B_pt_3);
  // y_tmp(3) = (contact_A_pt_4 - contact_B_pt_4).dot(contact_A_pt_4 -
  //            contact_B_pt_4);

  // *y = y_tmp;
}

void DummyConstraint::DoEval(const
                             Eigen::Ref<const VectorX<Variable>>& x,
                             VectorX<Expression>*y) const {
  throw std::logic_error(
    "TreeConstraint does not support symbolic evaluation.");
}




}  // namespace dairlib

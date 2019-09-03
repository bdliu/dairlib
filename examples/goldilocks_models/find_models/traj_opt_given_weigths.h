#pragma once

#include <string>
#include <Eigen/Dense>
#include "solvers/optimization_utils.h"
#include "systems/trajectory_optimization/dircon_position_data.h"
#include "systems/trajectory_optimization/dircon_distance_data.h"
#include "systems/trajectory_optimization/dircon_kinematic_data_set.h"
#include "systems/trajectory_optimization/dircon_opt_constraints.h"
#include "systems/trajectory_optimization/hybrid_dircon.h"

#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/constraint.h"
#include "drake/solvers/solve.h"

#include "multibody/multibody_utils.h"

#include "examples/goldilocks_models/find_models/goldilocks_model_traj_opt.h"

using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
using drake::solvers::SolutionResult;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

using drake::multibody::MultibodyPlant;
using drake::AutoDiffXd;

using dairlib::systems::trajectory_optimization::DirconAbstractConstraint;

using drake::math::RotationMatrix;
using drake::math::RollPitchYaw;

namespace dairlib {
namespace goldilocks_models  {

void trajOptGivenWeights(
  const MultibodyPlant<double> & plant,
  const MultibodyPlant<AutoDiffXd> & plant_autoDiff,
  int n_s, int n_sDDot, int n_tau, int n_feature_s, int n_feature_sDDot,
  MatrixXd B_tau,
  const VectorXd & theta_s, const VectorXd & theta_sDDot,
  double stride_length, double ground_incline, double duration, int max_iter,
  double major_optimality_tol, double major_feasibility_tol,
  vector<double> var_scale,
  std::string directory, std::string init_file, std::string prefix,
  /*vector<VectorXd> * w_sol_vec,
  vector<MatrixXd> * A_vec, vector<MatrixXd> * H_vec,
  vector<VectorXd> * y_vec,
  vector<VectorXd> * lb_vec, vector<VectorXd> * ub_vec,
  vector<VectorXd> * b_vec,
  vector<VectorXd> * c_vec,
  vector<MatrixXd> * B_vec,*/
  double Q_double, double R_double,
  double eps_reg,
  bool is_get_nominal,
  bool is_zero_touchdown_impact,
  bool extend_model,
  bool is_add_tau_in_cost,
  int batch,
  int robot_option);

void addRegularization(bool is_get_nominal, double eps_reg,
                       GoldilocksModelTrajOpt& gm_traj_opt);
void setInitialGuessFromFile(const VectorXd& w_sol,
                             GoldilocksModelTrajOpt& gm_traj_opt);
void augmentConstraintToFixThetaScaling(MatrixXd & B, MatrixXd & A,
                                        VectorXd & y, VectorXd & lb, VectorXd & ub,
                                        int n_s, int n_feature_s,
                                        const VectorXd & theta_s, int batch);

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

class LeftFootYConstraint : public DirconAbstractConstraint<double> {
 public:
  LeftFootYConstraint(const MultibodyPlant<double>* plant,
                      vector<double> var_scale) :
    DirconAbstractConstraint<double>(
      1, plant->num_positions(),
      VectorXd::Ones(1) * 0.03,
      VectorXd::Ones(1) * std::numeric_limits<double>::infinity(),
      "left_foot_constraint"),
    plant_(plant),
    body_(plant->GetBodyByName("toe_left")),
    quaternion_scale_(var_scale[4]) {
  }
  ~LeftFootYConstraint() override = default;

  void EvaluateConstraint(const Eigen::Ref<const drake::VectorX<double>>& x,
                          drake::VectorX<double>* y) const override {
    VectorXd q = x;
    q.head(4) *= quaternion_scale_;

    std::unique_ptr<drake::systems::Context<double>> context =
          plant_->CreateDefaultContext();
    plant_->SetPositions(context.get(), q);

    VectorX<double> pt(3);
    this->plant_->CalcPointsPositions(*context,
                                      body_.body_frame(), Vector3d::Zero(),
                                      plant_->world_frame(), &pt);
    *y = pt.segment(1, 1);
  };
 private:
  const MultibodyPlant<double>* plant_;
  const drake::multibody::Body<double>& body_;
  double quaternion_scale_;
};
class RightFootYConstraint : public DirconAbstractConstraint<double> {
 public:
  RightFootYConstraint(const MultibodyPlant<double>* plant,
                       vector<double> var_scale) :
    DirconAbstractConstraint<double>(
      1, plant->num_positions(),
      VectorXd::Ones(1) * (-std::numeric_limits<double>::infinity()),
      VectorXd::Ones(1) * (-0.03),
      "right_foot_constraint"),
    plant_(plant),
    body_(plant->GetBodyByName("toe_right")),
    quaternion_scale_(var_scale[4]) {
  }
  ~RightFootYConstraint() override = default;

  void EvaluateConstraint(const Eigen::Ref<const drake::VectorX<double>>& x,
                          drake::VectorX<double>* y) const override {
    VectorXd q = x;
    q.head(4) *= quaternion_scale_;

    std::unique_ptr<drake::systems::Context<double>> context =
          plant_->CreateDefaultContext();
    plant_->SetPositions(context.get(), q);

    VectorX<double> pt(3);
    this->plant_->CalcPointsPositions(*context,
                                      body_.body_frame(), Vector3d::Zero(),
                                      plant_->world_frame(), &pt);
    *y = pt.segment(1, 1);
  };
 private:
  const MultibodyPlant<double>* plant_;
  const drake::multibody::Body<double>& body_;
  double quaternion_scale_;
};

}  // namespace goldilocks_models
}  // namespace dairlib

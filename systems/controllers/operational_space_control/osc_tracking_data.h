#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "drake/multibody/rigid_body_tree.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/common/trajectories/exponential_plus_piecewise_polynomial.h"

#include "systems/framework/output_vector.h"

namespace dairlib {
namespace systems {
namespace controllers {

// OscTrackingData is a virtual class

/// Input of the constructor:
/// - dimension of the output/trajectory
/// - gains of PD controller
/// - cost weight
/// - a flag indicating the trajectory is a constant
/// - a flag indicating the trajecotry has exponential term (that is, the traj
///   is of ExponentialPlusPiecewisePolynomial class)

/// Cost:
///   0.5 * (J_*dv + JdotV - y_command)^T * W_ * (J_*dv + JdotV - y_command),
///   where dv is the decision variable of QP, and
///   y_commend = ddy_des_ + K_p_ * (error_y_) + K_d_ * (dy_des_ - J_ * v)
/// We ignore the cosntant term in cost function, since it doesn't affect
/// solution.

/// Most of the time, error_y_ = y_des_ - y_. The exception happends when we are
/// tracking rotation trajectory.

/// error_y_, JdotV and J_ are implemented in the derived class.

class OscTrackingData {
 public:
  OscTrackingData(std::string name, int n_r,
                  Eigen::MatrixXd K_p,
                  Eigen::MatrixXd K_d,
                  Eigen::MatrixXd W,
                  bool traj_is_const = false,
                  bool traj_has_exp = false);

  OscTrackingData() {}  // Default constructor

  // Updater and getters used by osc block
  bool Update(Eigen::VectorXd x,
              KinematicsCache<double>& cache,
              RigidBodyTree<double>* tree,
              const drake::trajectories::Trajectory<double>& traj, double t,
              int finite_state_machine_state,
              double time_since_last_state_switch);
  Eigen::VectorXd GetOutput() {return y_;}
  Eigen::VectorXd GetJ() {return J_;}
  Eigen::VectorXd GetJdotTimesV() {return JdotV_;}
  Eigen::VectorXd GetDesiredOutputWithPdControl(Eigen::VectorXd v);
  Eigen::MatrixXd GetWeight();
  bool TrajHasExp() {return traj_has_exp_;}

  // Getters
  std::string GetName() {return name_;};
  int GetTrajDim() {return n_r_;};
  bool TrajIsConst() {return traj_is_const_;}
  Eigen::VectorXd GetFixedPosition() {return fixed_position_;}

  // Setters
  void SetPGain(Eigen::MatrixXd K_p) {K_p_ = K_p;}
  void SetDGain(Eigen::MatrixXd K_d) {K_d_ = K_d;}
  void SetWeight(Eigen::MatrixXd W);
  // Set constant trajectory
  void SetConstantTraj(Eigen::VectorXd v);
  // No control peirod
  void SetNoControlPeriod(double duration) {period_of_no_control_ = duration;}

  // Run this function in OSC constructor to make sure that users constructed
  // OscTrackingData correctly.
  void CheckOscTrackingData();

 protected:
  int GetStateIdx() {return state_idx_;};

  // Feedback output, jacobian and dJ/dt * v
  Eigen::VectorXd error_y_;
  Eigen::VectorXd y_;
  Eigen::VectorXd ydot_;
  Eigen::MatrixXd J_;
  Eigen::VectorXd JdotV_;

  // Desired output
  Eigen::VectorXd y_des_;
  Eigen::VectorXd dy_des_;
  Eigen::VectorXd ddy_des_;

  // The states of finite state machine where the tracking is enabled
  // If `state_` is empty, then the tracking is always on.
  std::vector<int> state_;

 private:
  // Check if we should do tracking in the current state
  void TrackOrNot(int finite_state_machine_state,
                  double time_since_last_state_switch);

  // Updaters of feedback output, jacobian and dJ/dt * v
  virtual void UpdateError(const Eigen::VectorXd& x,
                           KinematicsCache<double>& cache,
                           RigidBodyTree<double>* tree) = 0;
  virtual void UpdateJ(const Eigen::VectorXd& x,
                       KinematicsCache<double>& cache,
                       RigidBodyTree<double>* tree) = 0;
  virtual void UpdateJdotV(const Eigen::VectorXd& x,
                           KinematicsCache<double>& cache,
                           RigidBodyTree<double>* tree) = 0;

  std::string name_;

  // dimension of the traj
  int n_r_;

  // PD control gains
  Eigen::MatrixXd K_p_;
  Eigen::MatrixXd K_d_;

  // Cost weights
  Eigen::MatrixXd W_;

  // Trajectory info
  bool traj_is_const_;
  bool traj_has_exp_;

  // Store the fixed position if the traj is a const
  Eigen::VectorXd fixed_position_;

  // A period when we don't apply control
  // (starting at the time when fsm switches to a new state)
  double period_of_no_control_ = 0;  // Unit: seconds

  // cache
  bool track_at_current_step_;
  int state_idx_;
};


class TaskSpaceTrackingData : public OscTrackingData {
 public:
  TaskSpaceTrackingData(std::string name, int n_r,
                        Eigen::MatrixXd K_p,
                        Eigen::MatrixXd K_d,
                        Eigen::MatrixXd W,
                        bool traj_is_const = false,
                        bool traj_has_exp = false);

  TaskSpaceTrackingData() {}  // Default constructor

  void AddPointToTrack(int body_index, Eigen::VectorXd pt_on_body, int state);
  void AddPointToTrack(std::vector<int> body_index,
                       std::vector<Eigen::VectorXd> pt_on_body,
                       std::vector<int> state);

 protected:
  std::vector<int> body_index_;
  std::vector<Eigen::VectorXd> pt_on_body_;

 private:
  virtual void UpdateError(const Eigen::VectorXd& x,
                           KinematicsCache<double>& cache,
                           RigidBodyTree<double>* tree);
  virtual void UpdateJ(const Eigen::VectorXd& x,
                       KinematicsCache<double>& cache,
                       RigidBodyTree<double>* tree);
  virtual void UpdateJdotV(const Eigen::VectorXd& x,
                           KinematicsCache<double>& cache,
                           RigidBodyTree<double>* tree);
};


class TransTaskSpaceTrackingData : public TaskSpaceTrackingData {
 public:
  TransTaskSpaceTrackingData(std::string name, int n_r,
                             Eigen::MatrixXd K_p,
                             Eigen::MatrixXd K_d,
                             Eigen::MatrixXd W,
                             bool traj_is_const = false,
                             bool traj_has_exp = false,
                             bool track_center_of_mass = false);

  TransTaskSpaceTrackingData() {}  // Default constructor

 private:
  void UpdateError(const Eigen::VectorXd& x,
                   KinematicsCache<double>& cache,
                   RigidBodyTree<double>* tree) final;
  void UpdateJ(const Eigen::VectorXd& x,
               KinematicsCache<double>& cache,
               RigidBodyTree<double>* tree) final;
  void UpdateJdotV(const Eigen::VectorXd& x,
                   KinematicsCache<double>& cache,
                   RigidBodyTree<double>* tree) final;
  bool track_center_of_mass_;
};


class RotTaskSpaceTrackingData : public TaskSpaceTrackingData {
 public:
  RotTaskSpaceTrackingData(std::string name, int n_r,
                           Eigen::MatrixXd K_p,
                           Eigen::MatrixXd K_d,
                           Eigen::MatrixXd W,
                           bool traj_is_const = false,
                           bool traj_has_exp = false,
                           Eigen::Isometry3d isometry = Eigen::Isometry3d::Identity());

  RotTaskSpaceTrackingData() {}  // Default constructor

 private:
  void UpdateError(const Eigen::VectorXd& x,
                   KinematicsCache<double>& cache,
                   RigidBodyTree<double>* tree) final;
  void UpdateJ(const Eigen::VectorXd& x,
               KinematicsCache<double>& cache,
               RigidBodyTree<double>* tree) final;
  void UpdateJdotV(const Eigen::VectorXd& x,
                   KinematicsCache<double>& cache,
                   RigidBodyTree<double>* tree) final;
  Eigen::Isometry3d isometry_;
};


class JointSpaceTrackingData : public OscTrackingData {
 public:
  JointSpaceTrackingData(std::string name, int n_r,
                         Eigen::MatrixXd K_p,
                         Eigen::MatrixXd K_d,
                         Eigen::MatrixXd W,
                         bool traj_is_const = false,
                         bool traj_has_exp = false);

  JointSpaceTrackingData() {}  // Default constructor

  void AddJointToTrack(int joint_pos_idx,
                       int joint_vel_idx,
                       int state);
  void AddJointToTrack(std::vector<int> joint_pos_idx,
                       std::vector<int> joint_vel_idx,
                       std::vector<int> state);

 private:
  void UpdateError(const Eigen::VectorXd& x,
                   KinematicsCache<double>& cache,
                   RigidBodyTree<double>* tree) final;
  void UpdateJ(const Eigen::VectorXd& x,
               KinematicsCache<double>& cache,
               RigidBodyTree<double>* tree) final;
  void UpdateJdotV(const Eigen::VectorXd& x,
                   KinematicsCache<double>& cache,
                   RigidBodyTree<double>* tree) final;

  std::vector<int> joint_pos_idx_;
  std::vector<int> joint_vel_idx_;
};


class AbstractTrackingData : public OscTrackingData {
 public:
  AbstractTrackingData(std::string name, int n_r,
                       Eigen::MatrixXd K_p,
                       Eigen::MatrixXd K_d,
                       Eigen::MatrixXd W,
                       bool traj_is_const = false,
                       bool traj_has_exp = false);

  AbstractTrackingData() {}  // Default constructor

 private:
  void UpdateError(const Eigen::VectorXd& x,
                   KinematicsCache<double>& cache,
                   RigidBodyTree<double>* tree) final;
  void UpdateJ(const Eigen::VectorXd& x,
               KinematicsCache<double>& cache,
               RigidBodyTree<double>* tree) final;
  void UpdateJdotV(const Eigen::VectorXd& x,
                   KinematicsCache<double>& cache,
                   RigidBodyTree<double>* tree) final;
};



}  // namespace controllers
}  // namespace systems
}  // namespace dairlib

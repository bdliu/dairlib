
#include <memory>
#include <chrono>

#include <string>

#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"

#include "drake/multibody/parsing/parser.h"
#include "drake/systems/rendering/multibody_position_to_geometry_pose.h"
#include "drake/geometry/geometry_visualization.h"

#include "common/find_resource.h"
#include "systems/primitives/subvector_pass_through.h"

#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/constraint.h"
#include "systems/trajectory_optimization/dircon_util.h"
#include "systems/trajectory_optimization/dircon_position_data.h"
#include "systems/trajectory_optimization/dircon_kinematic_data_set.h"
#include "systems/trajectory_optimization/dircon_opt_constraints.h"
#include "systems/trajectory_optimization/hybrid_dircon.h"

#include "multibody/multibody_utils.h"
#include "multibody/visualization_utils.h"

#include "systems/goldilocks_models/symbolic_manifold.h"
#include "systems/goldilocks_models/file_utils.h"

#include "examples/Goldilocks_models/kinematics_constraint.h"
#include "examples/Goldilocks_models/dynamics_constraint.h"

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
using std::vector;
using std::cout;
using std::endl;
using std::string;
using std::map;

using drake::multibody::MultibodyPlant;
using drake::geometry::SceneGraph;
using drake::multibody::Body;
using drake::multibody::Parser;
using drake::systems::rendering::MultibodyPositionToGeometryPose;


namespace dairlib {
namespace goldilocks_models {

using systems::trajectory_optimization::HybridDircon;
using systems::trajectory_optimization::DirconDynamicConstraint;
using systems::trajectory_optimization::DirconKinematicConstraint;
using systems::trajectory_optimization::DirconOptions;
using systems::trajectory_optimization::DirconKinConstraintType;
using systems::SubvectorPassThrough;

class GoldilcocksModelTrajOpt {
 public:
  GoldilcocksModelTrajOpt(
      int n_s_, int n_sDDot_, int n_feature_s_, int n_feature_sDDot_,
      VectorXd & theta_s_, VectorXd & theta_sDDot_,
      std::unique_ptr<HybridDircon<double>> dircon_in,
      const MultibodyPlant<AutoDiffXd> * plant,
      const std::vector<int>& num_time_samples);

  Eigen::VectorBlock<const VectorXDecisionVariable> reduced_model_state(
      int index, int n_s) const;


  std::unique_ptr<HybridDircon<double>> dircon;

  std::shared_ptr<KinematicsConstraint>  kinematics_constraint;
  std::shared_ptr<DynamicsConstraint>  dynamics_constraint;
  std::vector<Binding<Constraint>> kinematics_constraint_bindings;
  std::vector<Binding<Constraint>> dynamics_constraint_bindings;


 private:
  int num_knots_;
  VectorXDecisionVariable z_vars_;
  int n_s_;
  int n_sDDot_;
  int n_feature_s_;
  int n_feature_sDDot_;
  VectorXd theta_s_;
  VectorXd theta_sDDot_;
};

}  // namespace goldilocks_models
}  // namespace dairlib


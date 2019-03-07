
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
      std::unique_ptr<HybridDircon<double>> Dircon_traj_opt_in,
      const MultibodyPlant<double>& plant,
      int N);

  void solve();

  Eigen::VectorBlock<const VectorXDecisionVariable> reduced_model_state(
      int index) const;

  std::unique_ptr<HybridDircon<double>> Dircon_traj_opt;

 private:
  int num_knots_;
  int n_z_;
  VectorXDecisionVariable theta_vars_;
  VectorXDecisionVariable z_vars_;
  VectorXDecisionVariable placeholder_z_vars_;





};

}  // namespace goldilocks_models
}  // namespace dairlib


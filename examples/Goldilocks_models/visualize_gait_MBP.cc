#include <gflags/gflags.h>

#include <memory>
#include <chrono>

#include <string>

#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"

#include "drake/lcm/drake_lcm.h"

#include "drake/multibody/parsing/parser.h"
#include "drake/systems/rendering/multibody_position_to_geometry_pose.h"
#include "drake/geometry/geometry_visualization.h"

#include "common/find_resource.h"
#include "systems/primitives/subvector_pass_through.h"

#include "multibody/multibody_utils.h"
#include "multibody/visualization_utils.h"

#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/lcm/drake_lcm.h"

#include "drake/common/trajectories/piecewise_polynomial.h"

#include "systems/goldilocks_models/file_utils.h"

using drake::multibody::MultibodyPlant;
using drake::geometry::SceneGraph;
using drake::multibody::Body;
using drake::multibody::Parser;
using drake::systems::rendering::MultibodyPositionToGeometryPose;

using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Matrix3Xd;
using drake::trajectories::PiecewisePolynomial;
using drake::MatrixX;
using std::vector;
using std::shared_ptr;
using std::cout;
using std::endl;

namespace dairlib {

void visualizeGait() {
  // Create MBP
  drake::systems::DiagramBuilder<double> builder;
  MultibodyPlant<double> plant;
  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  Parser parser(&plant, &scene_graph);

  std::string full_name = FindResourceOrThrow(
                            "examples/Goldilocks_models/PlanarWalkerWithTorso.urdf");
  parser.AddModelFromFile(full_name);
  plant.AddForceElement<drake::multibody::UniformGravityFieldElement>(
    -9.81 * Eigen::Vector3d::UnitZ());
  plant.WeldFrames(
    plant.world_frame(), plant.GetFrameByName("base"),
    drake::math::RigidTransform<double>(Vector3d::Zero()).GetAsIsometry3());
  plant.Finalize();


  // Create a testing piecewise polynomial
  std::vector<double> T_breakpoint = {0, 2};
  std::vector<MatrixXd> Y(T_breakpoint.size(), MatrixXd::Zero(14, 1));

  MatrixXd x0 = MatrixXd::Zero(14, 1);
  x0 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  MatrixXd x1 = MatrixXd::Zero(14, 1);
  x1 << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  Y[0] = x0;
  Y[1] = x1;

  PiecewisePolynomial<double> pp_xtraj =
      PiecewisePolynomial<double>::FirstOrderHold(T_breakpoint, Y);

  // visualizer
  int n_loops = 1;
  multibody::connectTrajectoryVisualizer(&plant, &builder, &scene_graph,
                                         pp_xtraj);
  auto diagram = builder.Build();
  while (true)
    for (int i = 0; i < n_loops; i++) {
      drake::systems::Simulator<double> simulator(*diagram);
      simulator.set_target_realtime_rate(1);
      simulator.Initialize();
      simulator.StepTo(pp_xtraj.end_time());
    }


  return;
}
} // dairlib


int main() {

  dairlib::visualizeGait();

  return 0;
}


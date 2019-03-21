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
using std::string;
using std::to_string;

namespace dairlib {

void visualizeGait() {

  // parameters
  const string directory = "examples/Goldilocks_models/data/";
  int iter = 1;
  int batch = 2;

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

  // Read in trajecotry
  MatrixXd state_mat =
    goldilocks_models::readCSV(directory + to_string(iter) + string("_") + to_string(batch) + string("_state_at_knots.csv"));
  VectorXd time_mat =
    goldilocks_models::readCSV(directory + to_string(iter) + string("_") + to_string(batch) + string("_time_at_knots.csv"));

  // Create a testing piecewise polynomial
  std::vector<double> T_breakpoint;
  for(int i=0; i<time_mat.size(); i++)
    T_breakpoint.push_back(time_mat(i));
  std::vector<MatrixXd> Y;
  for(int i=0; i<time_mat.size(); i++)
    Y.push_back(state_mat.col(i));
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


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

DEFINE_int32(iter_start, 1, "The iter #");
DEFINE_int32(iter_end, -1, "The iter #");
DEFINE_int32(batch, 0, "The batch #");
DEFINE_double(realtime_factor, 1, "Rate of which the traj is played back");
DEFINE_int32(n_step, 3, "# of foot steps");

void swapTwoBlocks(MatrixXd * mat, int i_1, int j_1, int i_2, int j_2,
                   int n_row, int n_col) {
  MatrixXd temp_block1 = mat->block(i_1, j_1, n_row, n_col);
  MatrixXd temp_block2 = mat->block(i_2, j_2, n_row, n_col);
  mat->block(i_1, j_1, n_row, n_col) = temp_block2;
  mat->block(i_2, j_2, n_row, n_col) = temp_block1;
}

void visualizeGait(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // parameters
  int iter_start = FLAGS_iter_start;
  int iter_end = (FLAGS_iter_end >= FLAGS_iter_start) ?
                 FLAGS_iter_end : FLAGS_iter_start;
  int n_step = FLAGS_n_step;  // Should be > 0
  const string directory = "examples/Goldilocks_models/find_models/data/";

  // Looping through the iterations
  for (int iter = iter_start; iter <= iter_end; iter++) {
    // Read in trajecotry
    VectorXd time_mat =
      goldilocks_models::readCSV(directory + to_string(iter) + string("_") +
                                 to_string(FLAGS_batch) + string("_time_at_knots.csv"));
    MatrixXd state_mat =
      goldilocks_models::readCSV(directory + to_string(iter) + string("_") +
                                 to_string(FLAGS_batch) + string("_state_at_knots.csv"));

    int n_state = state_mat.rows();
    int n_q = n_state / 2;
    int n_knots = time_mat.rows();
    VectorXd ones = VectorXd::Ones(n_knots - 1);
    VectorXd xy_translation =
      state_mat.block(0, n_knots - 1, 2, 1) - state_mat.block(0, 0, 2, 1);

    // Concatenate the traj so it has multiple steps
    VectorXd time_mat_cat(n_step * n_knots - (n_step - 1));
    time_mat_cat(0) = 0;
    for (int i = 0; i < n_step; i++) {
      time_mat_cat.segment(1 + (n_knots - 1)*i, n_knots - 1) =
        time_mat.tail(n_knots - 1) + time_mat_cat((n_knots - 1) * i) * ones;
    }
    MatrixXd state_mat_cat(n_state, n_step * n_knots - (n_step - 1));
    state_mat_cat.col(0) = state_mat.col(0);
    for (int i = 0; i < n_step; i++) {
      state_mat_cat.block(0, 1 + (n_knots - 1)*i, n_state, n_knots - 1) =
        state_mat.block(0, 1, n_state, n_knots - 1);
      // Translate x and y
      for (int j = 0; j < 2; j++) {
        state_mat_cat.block(j, 1 + (n_knots - 1)*i, 1, n_knots - 1) =
          state_mat.block(j, 1, 1, n_knots - 1)  +
          i * xy_translation(j) * ones.transpose();
      }
      // Swap the leg
      if (i % 2) {
        // Position
        swapTwoBlocks(&state_mat_cat,
                      3, 1 + (n_knots - 1) * i,
                      4, 1 + (n_knots - 1) * i,
                      1, n_knots - 1);
        swapTwoBlocks(&state_mat_cat,
                      5, 1 + (n_knots - 1) * i,
                      6, 1 + (n_knots - 1) * i,
                      1, n_knots - 1);
        // Velocity (not necessary for plotting the traj)
        swapTwoBlocks(&state_mat_cat,
                      n_q + 3, 1 + (n_knots - 1) * i,
                      n_q + 4, 1 + (n_knots - 1) * i,
                      1, n_knots - 1);
        swapTwoBlocks(&state_mat_cat,
                      n_q + 5, 1 + (n_knots - 1) * i,
                      n_q + 6, 1 + (n_knots - 1) * i,
                      1, n_knots - 1);
      }
    }

    // Create a testing piecewise polynomial
    std::vector<double> T_breakpoint;
    for (int i = 0; i < time_mat_cat.size(); i++)
      T_breakpoint.push_back(time_mat_cat(i));
    std::vector<MatrixXd> Y;
    for (int i = 0; i < time_mat_cat.size(); i++)
      Y.push_back(state_mat_cat.col(i));
    PiecewisePolynomial<double> pp_xtraj =
      PiecewisePolynomial<double>::FirstOrderHold(T_breakpoint, Y);

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
      drake::math::RigidTransform<double>());
    plant.Finalize();

    // visualizer
    int n_loops = 1;
    multibody::connectTrajectoryVisualizer(&plant, &builder, &scene_graph,
                                           pp_xtraj);
    auto diagram = builder.Build();
    // while (true)
    for (int i = 0; i < n_loops; i++) {
      drake::systems::Simulator<double> simulator(*diagram);
      simulator.set_target_realtime_rate(FLAGS_realtime_factor);
      simulator.Initialize();
      simulator.StepTo(pp_xtraj.end_time());
    }
  }  // end for(int iter...)


  return;
}
} // dairlib

int main(int argc, char* argv[]) {

  dairlib::visualizeGait(argc, argv);

  return 0;
}


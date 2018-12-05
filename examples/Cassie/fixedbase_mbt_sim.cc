#include <memory>

#include <gflags/gflags.h>
#include "drake/lcm/drake_lcm.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/multibody/joints/floating_base_types.h" 
#include "drake/geometry/geometry_visualization.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/multibody/multibody_tree/joints/revolute_joint.h"
#include "drake/systems/analysis/runge_kutta2_integrator.h"

#include "systems/robot_lcm_systems.h"
#include "dairlib/lcmt_robot_output.hpp"
#include "dairlib/lcmt_robot_input.hpp"
#include "systems/primitives/subvector_pass_through.h"
#include "examples/Cassie/cassie_utils.h"

namespace dairlib {
using drake::systems::DiagramBuilder;
using drake::geometry::SceneGraph;
using drake::multibody::multibody_plant::MultibodyPlant;
using drake::systems::Context;
using drake::systems::Simulator;
using drake::multibody::RevoluteJoint;
using drake::systems::lcm::LcmSubscriberSystem;
using drake::systems::lcm::LcmPublisherSystem;
using dairlib::systems::SubvectorPassThrough;


// Simulation parameters.
DEFINE_double(target_realtime_rate, 1.0,  
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_bool(time_stepping, false, "If 'true', the plant is modeled as a "
    "discrete system with periodic updates. "
    "If 'false', the plant is modeled as a continuous system.");
DEFINE_double(dt, 1e-4, "The step size to use for compliant, ignored for time_stepping)");

int do_main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  DiagramBuilder<double> builder;

  drake::lcm::DrakeLcm lcm;

  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  scene_graph.set_name("scene_graph");

  const double time_step = FLAGS_time_stepping ? FLAGS_dt : 0.0;

  MultibodyPlant<double>& plant =
      *builder.AddSystem<MultibodyPlant>(time_step);
  addFixedBaseCassieMultibody(&plant, &scene_graph);

  // Create input receiver.
  auto input_sub = builder.AddSystem(
      LcmSubscriberSystem::Make<dairlib::lcmt_robot_input>("CASSIE_INPUT",
                                                           &lcm));
  auto input_receiver = builder.AddSystem<systems::RobotInputReceiver>(plant);
  builder.Connect(input_sub->get_output_port(),
                  input_receiver->get_input_port(0));

  // connect input receiver
  auto passthrough = builder.AddSystem<SubvectorPassThrough>(
    input_receiver->get_output_port(0).size(),
    0,
    plant.get_actuation_input_port().size());

  builder.Connect(input_receiver->get_output_port(0),
                  passthrough->get_input_port());
  builder.Connect(passthrough->get_output_port(),
                  plant.get_actuation_input_port());

  // Create state publisher.
  auto state_pub = builder.AddSystem(
      LcmPublisherSystem::Make<dairlib::lcmt_robot_output>("CASSIE_STATE",
                                                           &lcm));
  auto state_sender = builder.AddSystem<systems::RobotOutputSender>(plant);
  state_pub->set_publish_period(1.0/200.0);

  // connect state publisher
  builder.Connect(plant.get_continuous_state_output_port(),
                  state_sender->get_input_port_state());

  builder.Connect(state_sender->get_output_port(0),
                  state_pub->get_input_port());

  builder.Connect(
    plant.get_geometry_poses_output_port(),
    scene_graph.get_source_pose_port(plant.get_source_id().value()));

  builder.Connect(scene_graph.get_query_output_port(),
                  plant.get_geometry_query_input_port());

  auto diagram = builder.Build();


  // Create a context for this system:
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram_context->EnableCaching();
  diagram->SetDefaultContext(diagram_context.get());
  Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  plant.GetJointByName<RevoluteJoint>("hip_pitch_left").
      set_angle(&plant_context, .269);
  plant.GetJointByName<RevoluteJoint>("knee_left").
      set_angle(&plant_context, -.644);
  plant.GetJointByName<RevoluteJoint>("ankle_joint_left").
      set_angle(&plant_context, .792);
  plant.GetJointByName<RevoluteJoint>("toe_left").
      set_angle(&plant_context, -M_PI/3);

  plant.GetJointByName<RevoluteJoint>("hip_pitch_right").
      set_angle(&plant_context, .269);
  plant.GetJointByName<RevoluteJoint>("knee_right").
      set_angle(&plant_context, -.644);
  plant.GetJointByName<RevoluteJoint>("ankle_joint_right").
      set_angle(&plant_context, .792);
  plant.GetJointByName<RevoluteJoint>("toe_right").
      set_angle(&plant_context, -M_PI/3);



  Simulator<double> simulator(*diagram, std::move(diagram_context));

  if (!FLAGS_time_stepping) {
    // simulator.get_mutable_integrator()->set_maximum_step_size(0.01);
    // simulator.get_mutable_integrator()->set_target_accuracy(1e-1);
    // simulator.get_mutable_integrator()->set_fixed_step_mode(true);
    simulator.reset_integrator<drake::systems::RungeKutta2Integrator<double>>(
      *diagram, FLAGS_dt, &simulator.get_mutable_context());
  }

  simulator.set_publish_every_time_step(false);
  simulator.set_publish_at_initialization(false);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.StepTo(std::numeric_limits<double>::infinity());

  return 0;
}

}  // namespace dairlib

int main(int argc, char* argv[]) {
  return dairlib::do_main(argc, argv);
}

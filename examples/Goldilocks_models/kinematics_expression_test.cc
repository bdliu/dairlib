#include <iostream>
#include <string>
#include "math.h"
#include <Eigen/Dense>

#include "examples/Goldilocks_models/kinematics_expression.h"
#include "drake/common/drake_assert.h"


#include "drake/systems/framework/system.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/systems/rendering/multibody_position_to_geometry_pose.h"
#include "multibody/multibody_utils.h"
#include "common/find_resource.h"



using std::cout;
using std::endl;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;

using drake::AutoDiffVecXd;
using drake::AutoDiffXd;
using drake::math::DiscardGradient;
using drake::math::autoDiffToValueMatrix;
using drake::math::autoDiffToGradientMatrix;
using drake::math::initializeAutoDiff;

using dairlib::FindResourceOrThrow;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;

int main() {

  ////////////////////////////// Test 1 ////////////////////////////////////////
/*
  int n_z = 3;
  int n_x = 2;
  int n_feature = 5;
  dairlib::goldilocks_models::KinematicsExpression<double> expr_double(
    n_z, n_feature);
  dairlib::goldilocks_models::KinematicsExpression<AutoDiffXd> expr(
    n_z, n_feature);

  VectorXd x(n_x);
  // Matrix<double, Dynamic, 1> x(n_x);
  x << M_PI / 2, 3;
  AutoDiffVecXd x_autoDiff = initializeAutoDiff(x);
  DRAKE_DEMAND(n_x == x.size());

  ////// getFeature() //////
  VectorXd feature = expr.getFeature(x);
  // cout << "feature = \n" << feature << "\n\n";
  auto feature_autoDiff = expr.getFeature(x_autoDiff);
  cout << "feature_autoDiff = \n" << feature_autoDiff << "\n\n";

  ////// getDimFeature() //////
  // int num_feature = expr.getDimFeature();
  // cout << "num_feature = \n" << num_feature << "\n\n";
  // int num_feature_autoDiff = expr.getDimFeature();
  // cout << "num_feature_autoDiff = \n" << n_feature_autoDiff << "\n\n";

  ///// getExpression() //////
  VectorXd theta = VectorXd::Zero(n_z * n_feature);
  theta << 1, 1, 0, 0, 0,
           0, 0, 1, 0, 0,
           0, 0, 0, 1, 1;
  DRAKE_DEMAND(n_z * n_feature == theta.size());
  // Features implemented in KinematicsExpression should be:
  // feature << x(0),
  //            x(1)*x(1)*x(1),
  //            x(0) * x(1),
  //            cos(x(0)),
  //            sqrt(x(1));

  // expression =
  //      x(0) + x(1)*x(1)*x(1),
  //      x(0) * x(1),
  //      cos(x(0)) + sqrt(x(1));

  VectorX<double> expression_double = expr_double.getExpression(theta, x);
  cout << "double expression (double class) = \n" << expression_double << "\n\n";
  VectorX<double> expression = DiscardGradient(expr.getExpression(theta, x));
  cout << "double expression (AutoDiffXd class) = \n" << expression << "\n\n";

  AutoDiffVecXd theta_autoDiff =  initializeAutoDiff(theta);
  // auto expression_autoDiff = expr.getExpression(theta_autoDiff,x_autoDiff);
  auto expression_autoDiff = expr.getExpression(theta, x_autoDiff);
  cout << "expression_autoDiff = \n" << expression_autoDiff << "\n\n";

  // Checking autoDiff
  MatrixXd jacobian  = autoDiffToGradientMatrix(expression_autoDiff);
  cout << "jacobian = \n" << jacobian << "\n\n";
*/

  ////////////////////////////// Test 2 ////////////////////////////////////////
  // Get the position of the foot and check the gradient wrt state

  MultibodyPlant<double> plant;
  Parser parser(&plant);

  std::string full_name =
    FindResourceOrThrow("examples/Goldilocks_models/PlanarWalkerWithTorso.urdf");
  parser.AddModelFromFile(full_name);
  plant.AddForceElement<drake::multibody::UniformGravityFieldElement>(
    -9.81 * Eigen::Vector3d::UnitZ());
  plant.WeldFrames(
    plant.world_frame(), plant.GetFrameByName("base"),
    drake::math::RigidTransform<double>(Vector3d::Zero()).GetAsIsometry3());
  plant.Finalize();

  MultibodyPlant<AutoDiffXd> plant_autoDiff(plant);

  int n_z = 2; // Doesn't matter here
  int n_x = plant_autoDiff.num_positions() + plant_autoDiff.num_velocities() ;
  int n_feature = 3; // Doesn't matter here either actually
  dairlib::goldilocks_models::KinematicsExpression<double> expr_double(
    n_z, n_feature, &plant);
  dairlib::goldilocks_models::KinematicsExpression<AutoDiffXd> expr(
    n_z, n_feature, &plant_autoDiff);

  // Matrix<double, Dynamic, 1> x(n_x);
  VectorXd x = VectorXd::Zero(n_x);
  // x(3) = -M_PI/4;
  // x(5) =  M_PI/4;
  AutoDiffVecXd x_autoDiff = initializeAutoDiff(x);
  DRAKE_DEMAND(n_x == x.size());

  auto feature_autoDiff = expr.getFeature(x_autoDiff);
  cout << "feature_autoDiff = \n" << feature_autoDiff << "\n\n";

  // Checking autoDiff
  MatrixXd jacobian  = autoDiffToGradientMatrix(feature_autoDiff);
  cout << "jacobian = \n" << jacobian << "\n\n";



  return 0;
}
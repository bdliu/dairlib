#include <gflags/gflags.h>
#include <stdio.h>  // For removing files
#include <thread>  // multi-threading

#include "examples/Goldilocks_models/traj_opt_given_weigths.h"
#include "systems/goldilocks_models/file_utils.h"

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/solve.h"

#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/multibody/parsing/parser.h"

#include "common/find_resource.h"

#include "examples/Goldilocks_models/kinematics_expression.h"
#include "examples/Goldilocks_models/dynamics_expression.h"

using std::cin;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::to_string;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::VectorXcd;
using Eigen::MatrixXd;
using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;

using drake::geometry::SceneGraph;
using drake::multibody::MultibodyPlant;
using drake::multibody::Body;
using drake::multibody::Parser;
using drake::AutoDiffXd;
using dairlib::FindResourceOrThrow;

namespace dairlib {
namespace goldilocks_models {

DEFINE_int32(iter_start, 0, "The starting iteration #");
DEFINE_string(init_file, "w0.csv", "Initial Guess for Trajectory Optimization");
DEFINE_bool(is_newton, false, "Newton method or gradient descent");
DEFINE_bool(is_stochastic, true, "Random tasks or fixed tasks");
DEFINE_bool(is_debug, false, "Debugging or not");
DEFINE_bool(start_with_adjusting_stepsize, false, "");
DEFINE_bool(extend_model, false, "Extend the model in iteration # iter_start "
            "which is not equal to 0.");
DEFINE_int32(extend_model_iter, -1, "The starting iteration #");
DEFINE_bool(is_manual_initial_theta, false,
            "Assign initial theta of our choice");
DEFINE_bool(proceed_with_failure, false, "In the beginning, update theta even"
            "if there is a failed task");
DEFINE_bool(previous_iter_is_success, true, "Is the previous iter successful?");
DEFINE_bool(is_zero_touchdown_impact, false,
            "No impact force at fist touchdown");
DEFINE_bool(is_add_tau_in_cost, true, "Add RoM input in the cost function");
DEFINE_bool(is_multithread, false, "Use multi-thread or not");

DEFINE_int32(max_inner_iter, 500, "Max iteration # for traj opt");
DEFINE_int32(max_outer_iter, 10000, "Max iteration # for theta update");
DEFINE_double(h_step, 1e-4, "The step size for outer loop");
//                 // After adding tau
//                 // 1e-4 doesn't diverge
//                 // 1e-3 diverges
//                 // Before adding tau
//                 // 1e-3 is small enough to avoid gittering at the end
//                 // 1e-2 is a good compromise on both speed and gittering
//                 // 1e-1 caused divergence when close to optimal sol
DEFINE_double(eps_regularization, 1e-8, "Weight of regularization term"); //1e-4

MatrixXd solveInvATimesB(const MatrixXd & A, const MatrixXd & B) {
  MatrixXd X = (A.transpose() * A).ldlt().solve(A.transpose() * B);
  MatrixXd abs_resid = (A * X - B).cwiseAbs();
  VectorXd left_one = VectorXd::Ones(abs_resid.rows());
  VectorXd right_one = VectorXd::Ones(abs_resid.cols());
  cout << "sum-abs-residual: " << left_one.transpose()*abs_resid*right_one <<
       endl;
  return X;
}
// MatrixXd solveInvATimesB(const MatrixXd & A, const VectorXd & b) {
//   MatrixXd X = (A.transpose() * A).ldlt().solve(A.transpose() * b);
//   cout << "residual-norm: "<< (A*X-b).norm() << endl;
//   return X;
// }

int findGoldilocksModels(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Create MBP
  MultibodyPlant<double> plant;
  Parser parser(&plant);
  std::string full_name = FindResourceOrThrow(
                            "examples/Goldilocks_models/PlanarWalkerWithTorso.urdf");
  parser.AddModelFromFile(full_name);
  plant.AddForceElement<drake::multibody::UniformGravityFieldElement>(
    -9.81 * Eigen::Vector3d::UnitZ());
  plant.WeldFrames(
    plant.world_frame(), plant.GetFrameByName("base"),
    drake::math::RigidTransform<double>(Vector3d::Zero()).GetAsIsometry3());
  plant.Finalize();

  // Create autoDiff version of the plant
  MultibodyPlant<AutoDiffXd> plant_autoDiff(plant);

  // Random number generator
  std::random_device randgen;
  std::default_random_engine e1(randgen());
  std::random_device randgen2;
  std::default_random_engine e2(randgen2());

  // Files parameters
  const string directory = "examples/Goldilocks_models/data/";
  string init_file = FLAGS_init_file;
  // init_file = "w0_with_z.csv";
  string prefix = "";

  // Parametres for tasks (stride length and ground incline)
  cout << "\nTasks settings:\n";
  int N_sample_sl = 5;
  int N_sample_gi = 1;
  int N_sample = N_sample_sl * N_sample_gi; //1;
  double delta_stride_length = 0.03;
  double stride_length_0 = 0.3;
  double delta_ground_incline = 0.2;
  double ground_incline_0 = 0;
  double duration = 0.746; // Fix the duration now since we add cost ourselves
  cout << "N_sample_sl = " << N_sample_sl << endl;
  cout << "N_sample_gi = " << N_sample_gi << endl;
  cout << "delta_stride_length = " << delta_stride_length << endl;
  cout << "stride_length_0 = " << stride_length_0 << endl;
  cout << "delta_ground_incline = " << delta_ground_incline << endl;
  cout << "ground_incline_0 = " << ground_incline_0 << endl;

  // Paramters for the outer loop optimization
  cout << "\nOptimization setting:\n";
  int iter_start = FLAGS_iter_start;
  int max_outer_iter = FLAGS_max_outer_iter;
  double stopping_threshold = 1e-4;
  double h_step = FLAGS_h_step;
  double eps_regularization = FLAGS_eps_regularization;
  double indpt_row_tol = 1e-6;//1e-6
  bool is_newton = FLAGS_is_newton;
  bool is_stochastic = FLAGS_is_stochastic;
  is_newton ? cout << "Newton method\n" : cout << "Gradient descent method\n";
  is_stochastic ? cout << "Stocastic\n" : cout << "Non-stochastic\n";
  cout << "Step size = " << h_step << endl;
  cout << "eps_regularization = " << eps_regularization << endl;
  FLAGS_is_zero_touchdown_impact ? cout << "Zero touchdown impact\n" :
                                        cout << "Non-zero touchdown impact\n";

  // Paramters for the inner loop optimization
  int max_inner_iter = FLAGS_max_inner_iter;
  double R = 10;  // Cost on input effort
  double Q_double = 10; // Cost on velocity

  // Reduced order model parameters
  cout << "\nReduced-order model setting:\n";
  cout << "Warning: Need to make sure that the implementation in "
       "DynamicsExpression agrees with n_s and n_tau.\n";
  cout << "Warning: Need to make sure that you use the right initial theta.\n";
  int n_s = 2; //2
  int n_sDDot = n_s; // Assume that are the same (no quaternion)
  int n_tau = 1;
  cout << "n_s = " << n_s << ", n_tau = " << n_tau << endl;
  MatrixXd B_tau = MatrixXd::Zero(n_sDDot, n_tau);
  // B_tau = MatrixXd::Identity(2, 2);
  B_tau(1, 0) = 1;
  // B_tau(2, 1) = 1;
  // B_tau(0,0) = 1;
  // B_tau(2, 0) = 1;
  // B_tau(3, 1) = 1;
  cout << "B_tau = \n" << B_tau << endl;

  // Reduced order model setup
  KinematicsExpression<double> kin_expression(n_s, 0, &plant);
  DynamicsExpression dyn_expression(n_sDDot, 0);
  VectorXd dummy_q = VectorXd::Zero(plant.num_positions());
  VectorXd dummy_s = VectorXd::Zero(n_s);
  int n_feature_s = kin_expression.getFeature(dummy_q).size();
  int n_feature_sDDot =
    dyn_expression.getFeature(dummy_s, dummy_s).size();
  cout << "n_feature_s = " << n_feature_s << endl;
  cout << "n_feature_sDDot = " << n_feature_sDDot << endl;
  int n_theta_s = n_s * n_feature_s;
  int n_theta_sDDot = n_sDDot * n_feature_sDDot;
  VectorXd theta_s(n_theta_s);
  VectorXd theta_sDDot(n_theta_sDDot);

  // Initial guess of theta
  theta_s = VectorXd::Zero(n_theta_s);
  theta_sDDot = VectorXd::Zero(n_theta_sDDot);
  // theta_s(1) = 1;
  // theta_s(2 + n_feature_s) = 1;
  // theta_s(3 + 2 * n_feature_s) = 1;
  // theta_s(2) = 1; // LIPM
  // theta_sDDot(0) = 1;
  // // 2D LIPM
  // theta_s(0) = 1;
  // theta_s(1 + n_feature_s) = 1;
  // theta_sDDot(0) = 1;
  // // 2D LIPM with 2D swing foot
  theta_s(0) = 1;
  theta_s(1 + n_feature_s) = 1;
  // theta_s(2 + 2 *n_feature_s) = 1;
  // theta_s(3 + 3 * n_feature_s) = 1;
  theta_sDDot(0) = 1;
  // // Testing intial theta
  // theta_s = 0.25*VectorXd::Ones(n_theta_s);
  // theta_sDDot = 0.5*VectorXd::Ones(n_theta_sDDot);
  // theta_s = VectorXd::Random(n_theta_s);
  // theta_sDDot = VectorXd::Random(n_theta_sDDot);
  if (iter_start > 0) {
    if (!FLAGS_is_manual_initial_theta) {
      theta_s = readCSV(directory + to_string(iter_start) +
                        string("_theta_s.csv")).col(0);
      theta_sDDot = readCSV(directory + to_string(iter_start) +
                            string("_theta_sDDot.csv")).col(0);
    }
    else {
      MatrixXd theta_s_0_mat =
        readCSV(directory + string("theta_s_0.csv"));
      MatrixXd theta_sDDot_0_mat =
        readCSV(directory + string("theta_sDDot_0.csv"));
      theta_s.head(theta_s_0_mat.rows()) = theta_s_0_mat.col(0);
      theta_sDDot.head(theta_sDDot_0_mat.rows()) = theta_sDDot_0_mat.col(0);
    }
  }

  // Vectors/Matrices for the outer loop
  vector<VectorXd> w_sol_vec;
  vector<MatrixXd> H_vec;
  vector<VectorXd> b_vec;
  vector<VectorXd> c_vec;
  vector<MatrixXd> A_vec;
  vector<MatrixXd> A_active_vec;
  vector<VectorXd> lb_vec;
  vector<VectorXd> ub_vec;
  vector<VectorXd> y_vec;
  vector<VectorXd> y_active_vec;
  vector<MatrixXd> B_vec;
  vector<MatrixXd> B_active_vec;

  // Some setup
  cout << "\nOther settings:\n";
  double min_so_far;
  if (iter_start > 1  && !FLAGS_is_debug) {
    double old_cost = 0;
    for (int i = 0; i < N_sample; i++) {
      MatrixXd c = readCSV(directory + to_string(iter_start - 1) +  "_" +
                           to_string(i) + string("_c.csv"));
      old_cost += c(0, 0) / N_sample;
    }
    min_so_far = old_cost;
    cout << "min_so_far = " << min_so_far << endl;
  }
  else {
    min_so_far = 10000000;
  }
  // Tasks setup
  std::uniform_real_distribution<> dist_sl(
    -delta_stride_length / 2, delta_stride_length / 2);
  vector<double> delta_stride_length_vec;
  for (int i = 0 - N_sample_sl / 2; i < N_sample_sl - N_sample_sl / 2; i++)
    delta_stride_length_vec.push_back(i * delta_stride_length);
  std::uniform_real_distribution<> dist_gi(
    -delta_ground_incline / 2, delta_ground_incline / 2);
  vector<double> delta_ground_incline_vec;
  for (int i = 0 - N_sample_gi / 2; i < N_sample_gi - N_sample_gi / 2; i++)
    delta_ground_incline_vec.push_back(i * delta_ground_incline);
  // Some setup
  int n_theta = n_theta_s + n_theta_sDDot;
  VectorXd theta(n_theta);
  theta << theta_s, theta_sDDot;
  VectorXd prev_theta = theta;
  VectorXd step_direction;
  double current_iter_step_size = h_step;
  bool previous_iter_is_success = FLAGS_previous_iter_is_success;
  bool has_been_all_success;
  if (FLAGS_proceed_with_failure || (iter_start <= 1))
    has_been_all_success = false;
  else
    has_been_all_success = true;
  cout << "has_been_all_success = " << has_been_all_success << endl;

  bool start_with_adjusting_stepsize = FLAGS_start_with_adjusting_stepsize;
  if (start_with_adjusting_stepsize) {
    MatrixXd prev_theta_s_mat =
      readCSV(directory + to_string(iter_start - 1) + string("_theta_s.csv"));
    MatrixXd prev_theta_sDDot_mat =
      readCSV(directory + to_string(iter_start - 1) + string("_theta_sDDot.csv"));
    prev_theta << prev_theta_s_mat.col(0), prev_theta_sDDot_mat.col(0);

    MatrixXd step_direction_mat =
      readCSV(directory + to_string(iter_start - 1) + string("_step_direction.csv"));
    step_direction = step_direction_mat.col(0);

    // Below only works for Gradient Descent method (not Newton's method)
    current_iter_step_size = h_step / sqrt(step_direction.norm());  // Heuristic
    cout << "current_iter_step_size = " << current_iter_step_size << endl;
  }

  cout << endl;
  bool extend_model = FLAGS_extend_model;
  int extend_model_iter = (FLAGS_extend_model_iter == -1) ?
                          iter_start : FLAGS_extend_model_iter;
  if (extend_model) {
    int temp = (extend_model_iter == 0) ? 1 : extend_model_iter;
    cout << "\nWill extend the model at iteration # " << temp << " by ";
    VectorXd theta_s_append = readCSV(directory +
                                      string("theta_s_append.csv")).col(0);
    DRAKE_DEMAND(theta_s_append.rows() % n_feature_s == 0);
    int n_extend = theta_s_append.rows() / n_feature_s;
    cout << n_extend << " dimension.\n";

    cout << "Make sure that you include old and the new version of dynamics "
         "feasture.\nProceed? (Y/N)\n";
    char answer[1];
    cin >> answer;
    if (!((answer[0] == 'Y') || (answer[0] == 'y'))) {
      cout << "Ending the program.\n";
      return 0;
    } else {
      cout << "Start the iterating...\n";
    }
  }

  // Start the gradient descent
  int iter;
  for (iter = iter_start; iter <= max_outer_iter; iter++)  {
    cout << "*********** Iteration " << iter << " *************" << endl;
    if (iter != 0)
      cout << "theta_sDDot.head(10) = " << theta_sDDot.head(10).transpose() << endl;

    // setup for each iteration
    bool is_get_nominal = iter == 0 ? true : false;
    int n_sample = is_get_nominal ? 1 : N_sample;
    int max_inner_iter_pass_in = is_get_nominal ? 1000 : max_inner_iter;
    bool extend_model_this_iter =
      (extend_model && (iter == extend_model_iter) && !is_get_nominal) ?
      true : false;

    // store initial parameter values
    prefix = to_string(iter) +  "_";
    if (!FLAGS_is_debug) {
      writeCSV(directory + prefix + string("theta_s.csv"), theta_s);
      writeCSV(directory + prefix + string("theta_sDDot.csv"), theta_sDDot);
    }

    // Clear the vectors/matrices before trajectory optimization
    A_vec.clear();
    B_vec.clear();
    H_vec.clear();
    A_active_vec.clear();
    B_active_vec.clear();
    lb_vec.clear();
    ub_vec.clear();
    y_vec.clear();
    y_active_vec.clear();
    b_vec.clear();
    c_vec.clear();
    w_sol_vec.clear();

    // Run trajectory optimization for different tasks first
    bool samples_are_success = true;
    bool a_sample_is_success = false;
    if (start_with_adjusting_stepsize) {
      samples_are_success = false;
      start_with_adjusting_stepsize = false;
    } else {
      // Create vector of threads for multithreading
      std::vector<std::thread> threads;
      bool is_multithread = FLAGS_is_multithread;

      for (int sample = 0; sample < n_sample; sample++) {
        /// setup for each sample
        double stride_length = is_get_nominal ? stride_length_0 :
                               stride_length_0 + delta_stride_length_vec[sample % N_sample_sl];
        double ground_incline = is_get_nominal ? ground_incline_0 :
                                ground_incline_0 + delta_ground_incline_vec[sample / N_sample_sl];
        if (!is_get_nominal && is_stochastic) {
          stride_length += dist_sl(e1);
          ground_incline += dist_gi(e2);
        }

        prefix = to_string(iter) +  "_" + to_string(sample) + "_";
        string init_file_pass_in;
        if (is_get_nominal && previous_iter_is_success)
          init_file_pass_in = init_file;
        else if (is_get_nominal && !previous_iter_is_success)
          init_file_pass_in = string("0_0_w.csv");
        else if (!has_been_all_success && previous_iter_is_success) {
          init_file_pass_in = string("0_0_w.csv");  // Use nominal traj
          // init_file_pass_in = string("");  // No initial guess for the first iter
          // init_file_pass_in = string("w0.csv");  // w0 as initial guess for the first iter
        }
        else if (!has_been_all_success && !previous_iter_is_success)
          init_file_pass_in = to_string(iter) +  "_" +
                              to_string(sample) + string("_w.csv");
        else if (iter == 1)
          init_file_pass_in = string("0_0_w.csv");
        else
          init_file_pass_in = to_string(iter - 1) +  "_" +
                              to_string(sample) + string("_w.csv");

        //Testing
        if (FLAGS_is_debug) {
          // init_file_pass_in = to_string(iter) +  "_" +
          //                     to_string(sample) + string("_w.csv");//////////////////////////////////////////////////////////////////////////
          // init_file_pass_in = string("1_2_w.csv");
          // stride_length = 0.3;
          // init_file_pass_in = string("19_2_w.csv");
          // init_file_pass_in = string("1_0_w.csv");
          init_file_pass_in = string("1_0_w.csv");
        }

        // Trajectory optimization with fixed model paramters
        if (is_multithread) {
          cout << "add task to thread\n";
          threads.push_back(std::thread(trajOptGivenWeights,
                                        std::ref(plant), std::ref(plant_autoDiff),
                                        n_s, n_sDDot, n_tau,
                                        n_feature_s, n_feature_sDDot, B_tau,
                                        std::ref(theta_s), std::ref(theta_sDDot),
                                        stride_length, ground_incline,
                                        duration, max_inner_iter_pass_in,
                                        directory, init_file_pass_in, prefix,
                                        Q_double, R,
                                        eps_regularization,
                                        is_get_nominal,
                                        FLAGS_is_zero_touchdown_impact,
                                        extend_model_this_iter,
                                        FLAGS_is_add_tau_in_cost,
                                        sample));
          cout << "Finished adding task to thread\n";
        } else {
          trajOptGivenWeights(plant, plant_autoDiff,
                              n_s, n_sDDot, n_tau,
                              n_feature_s, n_feature_sDDot,
                              B_tau, theta_s, theta_sDDot,
                              stride_length, ground_incline,
                              duration, max_inner_iter_pass_in,
                              directory, init_file_pass_in, prefix,
                              Q_double, R,
                              eps_regularization,
                              is_get_nominal,
                              FLAGS_is_zero_touchdown_impact,
                              extend_model_this_iter,
                              FLAGS_is_add_tau_in_cost,
                              sample);
          prefix = to_string(iter) +  "_" + to_string(sample) + "_";
          int sample_success =
            (readCSV(directory + prefix + string("is_success.csv")))(0, 0);
          samples_are_success = (samples_are_success & (sample_success == 1));
          a_sample_is_success = (a_sample_is_success | (sample_success == 1));
          if ((has_been_all_success && !samples_are_success) || FLAGS_is_debug)
            break;
        }
      }  // for(int sample...)

      if (is_multithread) {
        for (int sample = 0; sample < n_sample; sample++) {
          threads[sample].join();
          // delete threads[sample];

          prefix = to_string(iter) +  "_" + to_string(sample) + "_";
          int sample_success =
            (readCSV(directory + prefix + string("is_success.csv")))(0, 0);
          samples_are_success = (samples_are_success & (sample_success == 1));
          a_sample_is_success = (a_sample_is_success | (sample_success == 1));
          // if ((has_been_all_success && !samples_are_success) || FLAGS_is_debug)
          //   break;
        }  // for(int sample...)
      }
    }
    if (FLAGS_is_debug) break;

    // Logic for how to iterate
    if (samples_are_success && !is_get_nominal) has_been_all_success = true;
    bool current_iter_is_success;
    if (!FLAGS_proceed_with_failure) {
      current_iter_is_success = samples_are_success;
    } else {
      current_iter_is_success = has_been_all_success ?
                                samples_are_success : a_sample_is_success;
    }
    previous_iter_is_success = current_iter_is_success;

    //
    if (is_get_nominal) {
      if (!current_iter_is_success)
        iter -= 1;
    } else if (extend_model_this_iter) {  // Extend the model
      VectorXd theta_s_append = readCSV(directory +
                                        string("theta_s_append.csv")).col(0);
      int n_extend = theta_s_append.rows() / n_feature_s;

      // update n_s and n_sDDot
      int old_n_s = n_s;
      n_s += n_extend;
      n_sDDot += n_extend;
      // update n_tau
      n_tau += n_extend;
      // update n_feature_sDDot
      int old_n_feature_sDDot = n_feature_sDDot;
      DynamicsExpression dyn_expression(n_sDDot, 0);
      VectorXd dummy_s = VectorXd::Zero(n_s);
      n_feature_sDDot = dyn_expression.getFeature(dummy_s, dummy_s).size();
      // update n_theta_s and n_theta_sDDot
      n_theta_s = n_s * n_feature_s;
      n_theta_sDDot = n_sDDot * n_feature_sDDot;
      n_theta = n_theta_s + n_theta_sDDot;
      cout << "Updated n_s = " << n_s << endl;
      cout << "Updated n_sDDot = " << n_sDDot << endl;
      cout << "Updated n_tau = " << n_tau << endl;
      cout << "Updated n_feature_sDDot = " << n_feature_sDDot << endl;
      cout << "Updated n_theta_s = " << n_theta_s << endl;
      cout << "Updated n_theta_sDDot = " << n_theta_sDDot << endl;
      cout << "Updated n_theta = " << n_theta << endl;

      // update B_tau
      MatrixXd B_tau_old = B_tau;
      B_tau.resize(n_sDDot, n_tau);
      B_tau = MatrixXd::Zero(n_sDDot, n_tau);
      B_tau.block(0, 0, B_tau_old.rows(), B_tau_old.cols()) = B_tau_old;
      B_tau.block(B_tau_old.rows(), B_tau_old.cols(), n_extend, n_extend) =
        MatrixXd::Identity(n_extend, n_extend);
      cout << "Updated B_tau = \n" << B_tau << endl;
      // update theta_s
      prefix = to_string(iter) +  "_";
      writeCSV(directory + prefix + string("theta_s (before extension).csv"),
               theta_s);
      MatrixXd theta_s_old = theta_s;
      theta_s.resize(n_theta_s);
      theta_s << theta_s_old, theta_s_append;
      // update theta_sDDot
      writeCSV(directory + prefix + string("theta_sDDot (before extension).csv"),
               theta_sDDot);
      MatrixXd theta_sDDot_old = theta_sDDot;
      theta_sDDot.resize(n_theta_sDDot);
      theta_sDDot = VectorXd::Zero(n_theta_sDDot);
      VectorXd new_idx = readCSV(directory +
                                 string("theta_sDDot_new_index.csv")).col(0);
      for (int i = 0; i < old_n_feature_sDDot; i++)
        for (int j = 0; j < old_n_s; j++)
          theta_sDDot(new_idx(i) + j * n_feature_sDDot) = theta_sDDot_old(
                i + j * old_n_feature_sDDot);
      // update theta
      theta.resize(n_theta);
      theta << theta_s, theta_sDDot;

      // Some setup
      cout << "Reset has_been_all_success to false, in case the next iter "
           "is infeasible.\n";
      prev_theta.resize(n_theta);
      prev_theta = theta;
      step_direction.resize(n_theta);
      min_so_far = 10000000;

      // So that we can re-run the current iter
      iter -= 1;
      has_been_all_success = false;
      previous_iter_is_success = false;

      // Never extend model again (we just extend it once)
      extend_model = false;
      continue;

    } else {  // Update parameters
      // Read in w_sol_vec, A_vec, H_vec, y_vec, lb_vec, ub_vec, b_vec, c_vec, B_vec;
      for (int sample = 0; sample < n_sample; sample++) {
        prefix = to_string(iter) +  "_" + to_string(sample) + "_";
        VectorXd success =
          readCSV(directory + prefix + string("is_success.csv")).col(0);
        if (success(0)) {
          w_sol_vec.push_back(readCSV(directory + prefix + string("w.csv")));
          A_vec.push_back(readCSV(directory + prefix + string("A.csv")));
          H_vec.push_back(readCSV(directory + prefix + string("H.csv")));
          y_vec.push_back(readCSV(directory + prefix + string("y.csv")));
          lb_vec.push_back(readCSV(directory + prefix + string("lb.csv")));
          ub_vec.push_back(readCSV(directory + prefix + string("ub.csv")));
          b_vec.push_back(readCSV(directory + prefix + string("b.csv")));
          c_vec.push_back(readCSV(directory + prefix + string("c.csv")));
          B_vec.push_back(readCSV(directory + prefix + string("B.csv")));

          bool rm = true;
          rm = (remove( (directory + prefix + string("A.csv")).c_str() ) == 0) & rm;
          rm = (remove( (directory + prefix + string("H.csv")).c_str() ) == 0) & rm;
          rm = (remove( (directory + prefix + string("y.csv")).c_str() ) == 0) & rm;
          rm = (remove( (directory + prefix + string("lb.csv")).c_str() ) == 0) & rm;
          rm = (remove( (directory + prefix + string("ub.csv")).c_str() ) == 0) & rm;
          rm = (remove( (directory + prefix + string("b.csv")).c_str() ) == 0) & rm;
          rm = (remove( (directory + prefix + string("B.csv")).c_str() ) == 0) & rm;
          if ( !rm )
            cout << "Error deleting files\n";
        }
      }

      int n_succ_sample = c_vec.size();

      // Get the total cost if it's successful
      double total_cost = 0;
      if (current_iter_is_success) {
        for (int sample = 0; sample < n_succ_sample; sample++)
          total_cost += c_vec[sample](0) / n_succ_sample;
        if (total_cost <= min_so_far) min_so_far = total_cost;
        cout << "total_cost = " << total_cost << " (min so far: " <<
             min_so_far << ")\n\n";
      }

      if (!current_iter_is_success) {
        iter -= 1;
        if (has_been_all_success) {
          current_iter_step_size = current_iter_step_size / 2;
          // if(current_iter_step_size<1e-5){
          //   cout<<"switch to the other method.";
          //   is_newton = !is_newton;
          // }
          cout << "Step size shrinks to " << current_iter_step_size <<
               ". Redo this iteration.\n\n";

          if (iter + 1 == iter_start)
            cout << "Step_direction might not have been defined yet. "
                 "Next line might give segmentation fault\n";
          // Descent
          theta = prev_theta + current_iter_step_size * step_direction;

          // Assign theta_s and theta_sDDot
          theta_s = theta.head(n_theta_s);
          theta_sDDot = theta.tail(n_theta_sDDot);
        }
      }
      else {
        // Extract active and independent constraints
        cout << "Extracting active and independent rows of A\n";
        vector<double> nw_vec;  // size of decision var of traj opt for all tasks
        vector<double> nl_vec;  // # of rows of active constraints for all tasks
        for (int sample = 0; sample < n_succ_sample; sample++) {
          DRAKE_ASSERT(b_vec[sample].cols() == 1);
          DRAKE_ASSERT(lb_vec[sample].cols() == 1);
          DRAKE_ASSERT(ub_vec[sample].cols() == 1);
          DRAKE_ASSERT(y_vec[sample].cols() == 1);
          DRAKE_ASSERT(w_sol_vec[sample].cols() == 1);

          int nt_i = B_vec[sample].cols();
          int nw_i = A_vec[sample].cols();
          nw_vec.push_back(nw_i);

          int nl_i = 0;
          double tol = 1e-4;
          for (int i = 0; i < y_vec[sample].rows(); i++) {
            if (y_vec[sample](i) >= ub_vec[sample](i) - tol ||
                y_vec[sample](i) <= lb_vec[sample](i) + tol)
              nl_i++;
          }

          MatrixXd A_active(nl_i, nw_i);
          MatrixXd B_active(nl_i, nt_i);
          VectorXd y_active(nl_i);

          nl_i = 0;
          for (int i = 0; i < y_vec[sample].rows(); i++) {
            if (y_vec[sample](i) >= ub_vec[sample](i) - tol ||
                y_vec[sample](i) <= lb_vec[sample](i) + tol) {
              A_active.row(nl_i) = A_vec[sample].row(i);
              B_active.row(nl_i) = B_vec[sample].row(i);
              y_active(nl_i) = y_vec[sample](i);
              nl_i++;
            }
          }




          cout << "\n (After extracting active constraints) Run traj opt to "
               "check if your quadratic approximation is correct\n";
          nl_i = A_active.rows();
          nw_i = A_active.cols();
          MathematicalProgram quadprog;
          auto w2 = quadprog.NewContinuousVariables(nw_i, "w2");
          quadprog.AddLinearConstraint( A_active,
                                        VectorXd::Zero(nl_i),
                                        VectorXd::Zero(nl_i),
                                        w2);
          quadprog.AddQuadraticCost(H_vec[sample], b_vec[sample], w2);
          const auto result2 = Solve(quadprog);
          auto solution_result2 = result2.get_solution_result();
          cout << solution_result2 << " | ";
          cout << "Cost:" << result2.get_optimal_cost() << " | ";
          if (result2.is_success()) {
            VectorXd w_sol_check = result2.GetSolution(
                                     quadprog.decision_variables());
            cout << "w_sol norm:" << w_sol_check.norm() << endl;
            // cout << "This should be zero\n" <<
            //      VectorXd::Ones(nl_i).transpose()*A_active*w_sol_check << endl;
            cout << "if this is not zero, then w=0 is not optimal: " <<
                 w_sol_check.transpose()*b_vec[sample] << endl;
          }







          // Only add the rows that are linearly independent
          // cout << "Start extracting independent rows of A\n";
          std::vector<int> full_row_rank_idx;
          full_row_rank_idx.push_back(0);
          for (int i = 1; i < nl_i; i++) {
            // Construct test matrix
            int n_current_rows = full_row_rank_idx.size();
            MatrixXd A_test(n_current_rows + 1, nw_i);
            for (unsigned int j = 0 ; j < full_row_rank_idx.size(); j++) {
              A_test.block(j, 0, 1, nw_i) =
                A_active.row(full_row_rank_idx[j]);
            }
            A_test.block(n_current_rows, 0, 1, nw_i) = A_active.row(i);

            // Perform svd to check rank
            Eigen::BDCSVD<MatrixXd> svd(A_test);
            // double sigular_value = svd.singularValues()(n_current_rows);
            if (svd.singularValues()(n_current_rows) > indpt_row_tol) {
              full_row_rank_idx.push_back(i);
            }
          }
          // cout << "Finished extracting independent rows of A\n\n";

          nl_i = full_row_rank_idx.size();
          nl_vec.push_back(nl_i);

          // Assign the rows
          MatrixXd A_full_row_rank(nl_i, nw_i);
          MatrixXd B_full_row_rank(nl_i, nt_i);
          VectorXd y_full_row_rank(nl_i);
          for (int i = 0; i < nl_i; i++) {
            A_full_row_rank.row(i) = A_active.row(full_row_rank_idx[i]);
            B_full_row_rank.row(i) = B_active.row(full_row_rank_idx[i]);
            y_full_row_rank(i) = y_active(full_row_rank_idx[i]);
          }

          A_active_vec.push_back(A_full_row_rank);
          B_active_vec.push_back(B_full_row_rank);
          y_active_vec.push_back(y_full_row_rank);

          cout << "A active and independent rows = " <<
               A_active_vec[sample].rows() << endl;
        }  // end extracting active and independent constraints
        cout << endl;



        // Reference for solving a sparse linear system
        // https://eigen.tuxfamily.org/dox/group__TopicSparseSystems.html
        // https://eigen.tuxfamily.org/dox/group__LeastSquares.html

        // Our calculation below is based on the fact that the H matrices are pd and
        // symmetric, so we check them here.
        // However, H turned out not to be psd, since we have timestep h as decision
        // variable. (It came from running cost. ~h*u'*R*u, etc)
        // Fixed it by adding running cost by hand (but the timestep is fixed now).
        // Now H is always pd because we also added a regularization term.
        /*cout << "Checking if H is pd and symmetric\n";
        for (int sample = 0; sample < n_succ_sample; sample++) {
          // Check if H is symmetric
          VectorXd One_w = VectorXd::Ones(nw_vec[sample]);
          double sum =
            One_w.transpose() * (H_vec[sample] - H_vec[sample].transpose()) * One_w;
          if (sum != 0) cout << "H is not symmetric\n";

          // Check if H is pd
          VectorXd eivals_real = H_vec[sample].eigenvalues().real();
          for (int i = 0; i < eivals_real.size(); i++) {
            if (eivals_real(i) <= 0)
              cout << "H is not positive definite (with e-value = "
                   << eivals_real(i) << ")\n";
          }
        }
        cout << "Finished checking\n\n";*/


        // Get w in terms of theta (Get P_i and q_i where w = P_i * theta + q_i)
        vector<MatrixXd> P_vec;
        vector<VectorXd> q_vec;
        P_vec.clear();
        q_vec.clear();
        // cout << "Getting P matrix and q vecotr\n";
        // Method 1: use optimization program to solve it???

        // Method 2: use schur complement (see notes)
        /*
        // This one requires the Hessian H to be pd.
        // Also, although this method might be more computationally efficient, the
        // accuracy is not as high as when we use inverse() directly. The reason is
        // that the condition number of A and invH is high, so AinvHA' makes it very
        // ill-conditioned.
        for (int sample = 0; sample < n_succ_sample; sample++) {
          MatrixXd AinvHA = A_active_vec[sample] * solveInvATimesB(
                              H_vec[sample], A_active_vec[sample].transpose());
          VectorXd invQc = solveInvATimesB(H_vec[sample], b_vec[sample]);
          MatrixXd E = solveInvATimesB(AinvHA, B_active_vec[sample]);
          VectorXd F = -solveInvATimesB(AinvHA, A_active_vec[sample] * invQc);
          // Testing
          Eigen::BDCSVD<MatrixXd> svd(AinvHA);
          cout << "AinvHA':\n";
          cout << "  biggest singular value is " << svd.singularValues()(0) << endl;
          cout << "  smallest singular value is "
                  << svd.singularValues().tail(1) << endl;
          cout << "The condition number of A and invH are large. That's why AinvHA'"
                  "is ill-conditioned.\n";
          // cout << "singular values are \n" << svd.singularValues() << endl;

          MatrixXd Pi = -solveInvATimesB(H_vec[sample],
                                         A_active_vec[sample].transpose() * E);
          VectorXd qi = -solveInvATimesB(H_vec[sample],
                                b_vec[sample] + A_active_vec[sample].transpose() * F);
          cout << "qi norm (this number should be close to 0) = "
               << qi.norm() << endl;
          P_vec.push_back(Pi);
          q_vec.push_back(qi);
        }*/

        // Method 3: use inverse() directly
        // H_ext = [H A'; A 0]
        for (int sample = 0; sample < n_succ_sample; sample++) {
          int nl_i = nl_vec[sample];
          int nw_i = nw_vec[sample];
          MatrixXd H_ext(nw_i + nl_i, nw_i + nl_i);
          H_ext.block(0, 0, nw_i, nw_i) = H_vec[sample];
          H_ext.block(0, nw_i, nw_i, nl_i) = A_active_vec[sample].transpose();
          H_ext.block(nw_i, 0, nl_i, nw_i) = A_active_vec[sample];
          H_ext.block(nw_i, nw_i, nl_i, nl_i) = MatrixXd::Zero(nl_i, nl_i);

          // Testing
          // Eigen::BDCSVD<MatrixXd> svd(H_vec[sample]);
          // cout << "H:\n";
          // cout << "  biggest singular value is " << svd.singularValues()(0) << endl;
          // cout << "  smallest singular value is "
          //         << svd.singularValues().tail(1) << endl;
          // // cout << "singular values are \n" << svd.singularValues() << endl;
          // // Testing
          if (sample == 0) {
            Eigen::BDCSVD<MatrixXd> svd_3(H_ext);
            cout << "H_ext:\n";
            cout << "  biggest singular value is " <<
                 svd_3.singularValues()(0) << endl;
            cout << "  smallest singular value is "
                 << svd_3.singularValues().tail(1) << endl;
          }
          // cout << "\nStart inverting the matrix.\n";
          MatrixXd inv_H_ext = H_ext.inverse();
          // cout << "Finsihed inverting the matrix.\n";
          // // Testing
          // Eigen::BDCSVD<MatrixXd> svd_5(inv_H_ext);
          // cout << "inv_H_ext:\n";
          // cout << "  biggest singular value is " << svd_5.singularValues()(0) << endl;
          // cout << "  smallest singular value is "
          //      << svd_5.singularValues().tail(1) << endl;

          MatrixXd inv_H_ext11 = inv_H_ext.block(0, 0, nw_i, nw_i);
          MatrixXd inv_H_ext12 = inv_H_ext.block(0, nw_i, nw_i, nl_i);

          MatrixXd Pi = -inv_H_ext12 * B_active_vec[sample];
          VectorXd qi = -inv_H_ext11 * b_vec[sample];
          P_vec.push_back(Pi);
          q_vec.push_back(qi);

          // Testing
          MatrixXd abs_Pi = Pi.cwiseAbs();
          VectorXd left_one = VectorXd::Ones(abs_Pi.rows());
          VectorXd right_one = VectorXd::Ones(abs_Pi.cols());
          // cout << "sum-abs-Pi: " <<
          //      left_one.transpose()*abs_Pi*right_one << endl;
          // cout << "sum-abs-Pi divide by m*n: " <<
          //      left_one.transpose()*abs_Pi*right_one / (abs_Pi.rows()*abs_Pi.cols())
          //      << endl;
          double max_Pi_element = abs_Pi(0, 0);
          for (int i = 0; i < abs_Pi.rows(); i++)
            for (int j = 0; j < abs_Pi.cols(); j++) {
              if (abs_Pi(i, j) > max_Pi_element) max_Pi_element = abs_Pi(i, j);
            }
          cout << "max element of abs-Pi = " << max_Pi_element << endl;
          cout << "qi norm (this number should be close to 0) = "
               << qi.norm() << endl;
        }





        // // Testing
        // MatrixXd invH = H_vec[0].inverse();
        // Eigen::BDCSVD<MatrixXd> svd_4(invH);
        // cout << "invH:\n";
        // cout << "  biggest singular value is " << svd_4.singularValues()(0) << endl;
        // cout << "  smallest singular value is "
        //         << svd_4.singularValues().tail(1) << endl;
        // // Testing
        // Eigen::BDCSVD<MatrixXd> svd_2(A_active_vec[0]);
        // cout << "A:\n";
        // cout << "  biggest singular value is " << svd_2.singularValues()(0) << endl;
        // cout << "  smallest singular value is "
        //         << svd_2.singularValues().tail(1) << endl;
        // // cout << "singular values are \n" << svd_2.singularValues() << endl;





        // Run a quadprog to check if the solution to the following problem is 0
        // Theoratically, it should be 0. Otherwise, something is wrong
        // min 0.5*w^T Q w + c^T w
        // st  A w = 0
        /*cout << "Run traj opt to check if your quadratic approximation is correct\n";
        int nl_i = nl_vec[0];
        int nw_i = nw_vec[0];
        MathematicalProgram quadprog;
        auto w = quadprog.NewContinuousVariables(nw_i, "w");
        quadprog.AddLinearConstraint( A_active_vec[0],
                                  VectorXd::Zero(nl_i),
                                  VectorXd::Zero(nl_i),
                                  w);
        quadprog.AddQuadraticCost(H_vec[0],b_vec[0],w);
        const auto result = Solve(quadprog);
        auto solution_result = result.get_solution_result();
        cout << solution_result << endl;
        cout << "Cost:" << result.get_optimal_cost() << endl;
        VectorXd w_sol = result.GetSolution(quadprog.decision_variables());
        cout << "w_sol norm:" << w_sol.norm() << endl;
        cout << "Finished traj opt\n\n";*/







        /*// Check if Q_theta is pd
        cout << "Checking if Q_theta is psd...\n";
        MatrixXd Q_theta = MatrixXd::Zero(n_theta, n_theta);
        for (int sample = 0; sample < n_succ_sample; sample++)
          Q_theta += P_vec[sample].transpose()*H_vec[sample]*P_vec[sample];
        VectorXd eivals_real = Q_theta.eigenvalues().real();
        for (int i = 0; i < eivals_real.size(); i++) {
          if (eivals_real(i) <= 0)
            cout << "Q_theta is not positive definite (with e-value = "
                 << eivals_real(i) << ")\n";
        }
        cout << endl;*/







        // Get gradient of the cost wrt theta (assume H_vec[sample] symmetric)
        // cout << "Calculating gradient\n";
        VectorXd gradient_cost = VectorXd::Zero(theta.size());
        for (int sample = 0; sample < n_succ_sample; sample++) {
          gradient_cost += P_vec[sample].transpose() * b_vec[sample] / n_succ_sample;
          // gradient_cost +=
          // P_vec[sample].transpose() * (b_vec[sample] + H_vec[sample] * q_vec[sample]);
        }
        // cout << "gradient_cost = \n" << gradient_cost;

        // Newton's method (not exactly the same, cause Q_theta is not pd but psd)
        // See your IOE611 lecture notes on page 7-17 to page 7-20
        // cout << "Getting Newton step\n";
        MatrixXd Q_theta = MatrixXd::Zero(n_theta, n_theta);
        for (int sample = 0; sample < n_succ_sample; sample++) {
          Q_theta +=
            P_vec[sample].transpose() * H_vec[sample] * P_vec[sample] / n_succ_sample;
        }
        double mu = 1e-4; // 1e-6 caused unstable and might diverge
        MatrixXd inv_Q_theta = (Q_theta + mu * MatrixXd::Identity(n_theta,
                                n_theta)).inverse();
        VectorXd newton_step = -inv_Q_theta * gradient_cost;
        // Testing
        /*Eigen::BDCSVD<MatrixXd> svd(inv_Q_theta);
        cout << "inv_Q_theta's smallest and biggest singular value " <<
             svd.singularValues().tail(1) << ", " <<
             svd.singularValues()(0) << endl;*/
        // Newton decrement (can be a criterion to terminate your newton steps)
        double lambda_square = -gradient_cost.transpose() * newton_step;

        // step_direction
        step_direction = is_newton ? newton_step : -gradient_cost;
        prefix = to_string(iter) +  "_";
        writeCSV(directory + prefix + string("step_direction.csv"), step_direction);


        // Calculate lambda and gradient norm
        VectorXd lambda_square_vecXd(1); lambda_square_vecXd << lambda_square;
        VectorXd norm_grad_cost(1); norm_grad_cost << gradient_cost.norm();
        writeCSV(directory + prefix + string("norm_grad_cost.csv"), norm_grad_cost);
        writeCSV(directory + prefix + string("lambda_square.csv"), lambda_square_vecXd);
        cout << "lambda_square = " << lambda_square << endl;
        cout << "gradient_cost norm: " << norm_grad_cost << endl << endl;

        // Gradient descent
        prev_theta = theta;
        // current_iter_step_size = h_step;
        current_iter_step_size = h_step / sqrt(norm_grad_cost(0));  // Heuristic
        if (is_newton)
          theta = theta + current_iter_step_size * step_direction;
        else
          theta = theta + current_iter_step_size * step_direction;

        // Assign theta_s and theta_sDDot
        theta_s = theta.head(n_theta_s);
        theta_sDDot = theta.tail(n_theta_sDDot);

        // Check optimality
        if (is_newton) {
          if (lambda_square < stopping_threshold) {
            cout << "Found optimal theta.\n\n";
            break;
          }
        }
        else {
          if (norm_grad_cost(0) < stopping_threshold) {
            cout << "Found optimal theta.\n\n";
            break;
          }
        }
      }  // end if goes goes down
    }  // end if(!is_get_nominal)
  }  // end for

  // store parameter values
  prefix = to_string(iter + 1) +  "_";
  if (!FLAGS_is_debug) {
    writeCSV(directory + prefix + string("theta_s.csv"), theta_s);
    writeCSV(directory + prefix + string("theta_sDDot.csv"), theta_sDDot);
  }

  return 0;
}  // int findGoldilocksModels

}  // namespace goldilocks_models
}  // namespace dairlib

int main(int argc, char* argv[]) {
  return dairlib::goldilocks_models::findGoldilocksModels(argc, argv);
}

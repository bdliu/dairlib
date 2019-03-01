#include "examples/Goldilocks_models/goldilocks_model_traj_opt.h"


namespace dairlib {
namespace goldilocks_models {

  GoldilcocksModelTrajOpt::GoldilcocksModelTrajOpt(
      std::unique_ptr<HybridDircon<double>> Dircon_traj_opt_in,
      int N){
    Dircon_traj_opt = std::move(Dircon_traj_opt_in);
    num_knots_ = N;
  }


  // https://github.com/RobotLocomotion/drake/blob/master/systems/trajectory_optimization/multiple_shooting.cc
  // https://drake.mit.edu/doxygen_cxx/classdrake_1_1systems_1_1trajectory__optimization_1_1_multiple_shooting.html#a3d57e7972ccf310e19d3b73cac1c2a8c


  // Inside construct:(
                  // pass in the multipleShooting class
                  // pass in the number of knots
                  // pass in z constraint class
                  // pass in zdot constraint class
                  //)

    // add new decision variable z
    // In for loop
      // add constraint for z
      // (decision variables passed into the constraint is {x_i,z_i})
    // In for loop
      // You will need to do direct collocation for zDot (cubic spline)
      // 1. Get the spline from z0,z1,
      //    zDot0(functino of z0),zDot1(function of z1)
      // 2. The constraint is that at the middle point, the slope still match


  // methods
    // public:
      // solve()
        // solve MultipleShooting and then return the solution
    // private:
      // functions related to placeholder_z_vars_


  // members:
    // public:
      // DIRCON_traj_opt
    // privite:
      // z_vars_
      // placeholder_z_vars_







}  // namespace goldilocks_models
}  // namespace dairlib


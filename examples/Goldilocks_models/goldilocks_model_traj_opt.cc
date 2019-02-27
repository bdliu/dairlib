#include "examples/Goldilocks_models/goldilocks_model_traj_opt.h"


namespace dairlib {
namespace goldilocks_models {





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
  
    // what about zdot?


  // method1: solve()
    // solve MultipleShooting and then return the solution 
  // method2: 


  // members:
    // z_vars_
    // placeholder_z_vars_








}  // namespace goldilocks_models
}  // namespace dairlib


#include "examples/Goldilocks_models/dynamics_constraint.h"


namespace dairlib {
namespace goldilocks_models {

DynamicsConstraint::DynamicsConstraint(
                                 int n_zDot, int n_featureDot, int n_thetaDot,
                                 const MultibodyPlant<double>& plant,
                                 const std::string& description):
  Constraint(n_zDot,
             2*n_zDot + n_thetaDot + 1,
             VectorXd::Zero(n_zDot),
             VectorXd::Zero(n_zDot),
             description),
  plant_(plant),
  n_constraint_(n_zDot),
  n_featureZDot_(n_featureDot),
  n_thetaZDot_(n_thetaDot),
  expression_object_(DynamicsExpression(n_zDot, n_featureDot)) {

  // Check the theta size
  DRAKE_DEMAND((n_zDot/2) * n_featureDot == n_thetaDot);

  // Check the feature size implemented in the model expression
  VectorXd z_temp = VectorXd::Zero(n_zDot);
  DRAKE_DEMAND(n_featureDot == expression_object_.getFeature(z_temp).size());
}


void DynamicsConstraint::DoEval(const
                             Eigen::Ref<const Eigen::VectorXd>& q,
                             Eigen::VectorXd* y) const {
  AutoDiffVecXd y_t;
  Eval(initializeAutoDiff(q), &y_t);
  *y = autoDiffToValueMatrix(y_t);
}

void DynamicsConstraint::DoEval(const
                             Eigen::Ref<const AutoDiffVecXd>& q,
                             AutoDiffVecXd* y) const {


  // TODO: Need to update this



  *y = initializeAutoDiff(VectorXd::Zero(4));

}

void DynamicsConstraint::DoEval(const
                             Eigen::Ref<const VectorX<Variable>>& x,
                             VectorX<Expression>*y) const {
  throw std::logic_error(
    "This constraint class does not support symbolic evaluation.");
}




}  // namespace goldilocks_models
}  // namespace dairlib

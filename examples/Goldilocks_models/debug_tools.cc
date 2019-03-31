#include "examples/Goldilocks_models/debug_tools.h"

namespace dairlib {
namespace goldilocks_models  {

// Create time knots for creating cubic splines
vector<double> createTimeKnotsGivenTimesteps(vector<VectorXd> h_vec){
  vector<double> T_breakpoint;
  double time = 0;
  T_breakpoint.push_back(time);
  for (unsigned int i = 0; i < h_vec.size() ; i++) {
    time += h_vec[i](0);
    T_breakpoint.push_back(time);
  }
  return T_breakpoint;
}


PiecewisePolynomial<double> createCubicSplineGivenSAndSdot(vector<VectorXd> h_vec,
                                   vector<VectorXd> s_vec,
                                   vector<VectorXd> ds_vec) {
  // Create time knots
  vector<double> T_breakpoint = createTimeKnotsGivenTimesteps(h_vec);

  // Create traj value and its derivatives (convert VectorXd to MatrixXd)
  vector<MatrixXd> s(T_breakpoint.size(), MatrixXd::Zero(1, 1));
  vector<MatrixXd> s_dot(T_breakpoint.size(), MatrixXd::Zero(1, 1));
  for (unsigned int i = 0; i < s_vec.size() ; i++) {
    s[i] = s_vec[i];
    s_dot[i] = ds_vec[i];
  }

  // Construct splines
  return PiecewisePolynomial<double>::Cubic(T_breakpoint, s, s_dot);
}


void storeSplineOfS(vector<VectorXd> h_vec,
    PiecewisePolynomial<double> s_spline,
    string directory, string prefix){
  // parameters
  int n_sample_each_seg = 3;

  // setup
  int n_s = s_spline.value(0).rows();

  // Create time knots
  vector<double> T_breakpoint = createTimeKnotsGivenTimesteps(h_vec);

  // Create the matrix for csv file
  MatrixXd s_over_time(n_s, 1 + (n_sample_each_seg - 1)*h_vec.size());
  // cout << "s_over_time.rows() = " << s_over_time.rows() << endl;
  // cout << "s_over_time.cols() = " << s_over_time.cols() << endl;
  s_over_time.block(0, 0, n_s, 1) = s_spline.value(0.0);
  for (unsigned int i = 0; i < h_vec.size() ; i++) {
    for (int j = 1; j < n_sample_each_seg; j++) {
      // cout << j + i * (n_sample_each_seg - 1) << endl;
      s_over_time.block(0, j + i * (n_sample_each_seg - 1), n_s, 1) =
        s_spline.value(
          T_breakpoint[i] + j * h_vec[i](0) / (n_sample_each_seg - 1));
    }
  }
  // cout << "s_over_time = \n" << s_over_time << endl;

  // Store into csv file
  writeCSV(directory + prefix + string("s.csv"), s_over_time);
}


void checkSplineOfS(vector<VectorXd> h_vec, vector<VectorXd> dds_vec,
    PiecewisePolynomial<double> s_spline){
  // parameters
  double tol = 1e-4;

  // Create time knots
  vector<double> T_breakpoint = createTimeKnotsGivenTimesteps(h_vec);

  // Compare
  for (unsigned int i = 0; i < T_breakpoint.size() ; i++) {
    VectorXd dds_by_drake = s_spline.derivative(2).value(T_breakpoint[i]);
    VectorXd dds_by_hand = dds_vec[i];
    DRAKE_DEMAND((dds_by_drake - dds_by_hand).norm() <= tol);
  }

}


}  // namespace goldilocks_models
} // dairlib


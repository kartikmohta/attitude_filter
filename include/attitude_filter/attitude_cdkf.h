#ifndef ATTITUDE_CDKF_H
#define ATTITUDE_CDKF_H

#include <Eigen/Geometry>
#include <ros/ros.h>
#include "attitude_filter/attitude_state.h"

class AttitudeCDKF
{
 public:
  using Scalar_t = double;

  template <int N>
  using Vec = Eigen::Matrix<Scalar_t, N, 1>;

  template <int N>
  using RowVec = Eigen::Matrix<Scalar_t, 1, N>;

  template <int N, int M>
  using Mat = Eigen::Matrix<Scalar_t, N, M>;

  // Dimensions
  static constexpr int state_count_ = AttitudeState<Scalar_t>::state_count_;
  static constexpr int proc_noise_count_ = 6;
  static constexpr int input_count_ = 3;
  static constexpr int meas_accel_count_ = 3;

  // using OutputStateVec = Vec<output_state_count_>;
  using State_t = AttitudeState<Scalar_t>;
  using StateVec = State_t::StateVec;
  using StateCov = State_t::StateCov;
  using InputVec = Vec<input_count_>;
  using ProcNoiseVec = Vec<proc_noise_count_>;
  using ProcNoiseCov = Mat<proc_noise_count_, proc_noise_count_>;
  using MeasAccelVec = Vec<meas_accel_count_>;
  using MeasAccelCov = Mat<meas_accel_count_, meas_accel_count_>;

  AttitudeCDKF();

  // OutputStateVec GetState();
  State_t GetState();
  const StateCov &GetStateCovariance();
  const ros::Time &GetStateTime();

  void SetGravity(Scalar_t g);
  void SetParameter(Scalar_t h);
  void SetImuCovariance(const ProcNoiseCov &Rv);

  bool ProcessUpdate(const InputVec &u, const ros::Time &time);
  bool MeasurementUpdateAccel(const MeasAccelVec &z,
                              const MeasAccelCov &RnAccel,
                              const ros::Time &time);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

 private:
  // Private functions
  void GenerateWeights(unsigned int L);

  /* Generate the sigma points given the noise covariance matrix
   * param[in] Rn Noise covariance matrix used for generating the sigma points
   */
  template <typename T>
  Mat<state_count_ + T::RowsAtCompileTime,
      2 * (state_count_ + T::RowsAtCompileTime) + 1>
  GenerateSigmaPoints(const T &Rn)
  {
    constexpr int noise_count = T::RowsAtCompileTime;
    constexpr int L = state_count_ + noise_count;

    GenerateWeights(L);

    // Expand state
    Vec<L> xaa = Vec<L>::Zero();
    // xaa.template topRows<state_count_>() = xa_;
    Mat<L, L> Paa = Mat<L, L>::Zero();
    Paa.template block<state_count_, state_count_>(0, 0) = Pa_;
    Paa.template block<noise_count, noise_count>(state_count_, state_count_) =
        Rn;

    // Matrix square root
    Mat<L, L> sqrtPaa = Paa.llt().matrixL();

    Mat<L, 2 *L + 1> Xaa = xaa.template replicate<1, 2 * L + 1>();
    Xaa.template block<L, L>(0, 1).noalias() += h_ * sqrtPaa;
    Xaa.template block<L, L>(0, L + 1).noalias() -= h_ * sqrtPaa;
    return Xaa;
  }

  StateVec ProcessModel(const StateVec &dx, const InputVec &u,
                        const ProcNoiseVec &v, double dt);
  MeasAccelVec MeasurementModelAccel(const StateVec &dx);

  // State
  State_t state_;
  // StateVec xa_;
  StateCov Pa_;
  ros::Time prev_proc_update_time_;

  // Initial process update indicator
  bool init_process_;
  bool init_meas_accel_;

  // Process Covariance Matrix
  ProcNoiseCov Rv_;

  // Gravity
  Scalar_t g_;

  // CDKF Parameter
  Scalar_t h_;

  // CDKF Weights
  Scalar_t wm0_, wm1_;
  Scalar_t wc1_, wc2_;
};

#endif

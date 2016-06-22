#include "attitude_filter/attitude_cdkf.h"
#include <Eigen/Cholesky>
#include <Eigen/Geometry>

using Scalar_t = AttitudeCDKF::Scalar_t;

template <int N>
using Vec = AttitudeCDKF::Vec<3>;

template <int N, int M>
using Mat = AttitudeCDKF::Mat<N, M>;

using Quat = Eigen::Quaternion<Scalar_t>;


AttitudeCDKF::AttitudeCDKF()
{
  Pa_.setZero();
  Pa_(0, 0) = 10 * M_PI / 180 * 10 * M_PI / 180;
  Pa_(1, 1) = 10 * M_PI / 180 * 10 * M_PI / 180;
  Pa_(2, 2) = 10 * M_PI / 180 * 10 * M_PI / 180;
  Pa_(3, 3) = 0.01 * 0.01;
  Pa_(4, 4) = 0.01 * 0.01;
  Pa_(5, 5) = 0.01 * 0.01;

  Rv_.setIdentity();

  h_ = std::sqrt(3);
  g_ = 9.81;
  init_process_ = false;
  init_meas_accel_ = false;
}

AttitudeCDKF::State_t AttitudeCDKF::GetState()
{
  return state_;
}

const AttitudeCDKF::StateCov &AttitudeCDKF::GetStateCovariance()
{
  return Pa_;
}

void AttitudeCDKF::SetGravity(double g)
{
  g_ = g;
}

void AttitudeCDKF::SetImuCovariance(const ProcNoiseCov &Rv)
{
  Rv_ = Rv;
}

void AttitudeCDKF::SetParameter(double h)
{
  h_ = h;
}

bool AttitudeCDKF::ProcessUpdate(const InputVec &u, const ros::Time &time)
{
  // Init Time
  if(!init_meas_accel_)
  {
    prev_proc_update_time_ = time;
    return false;
  }

  double dt = (time - prev_proc_update_time_).toSec();
  prev_proc_update_time_ = time;

  constexpr unsigned int L = state_count_ + proc_noise_count_;

  // Generate sigma points
  const Mat<L, 2 *L + 1> Xaa = GenerateSigmaPoints(Rv_);

  // Extract state and noise from augmented state
  Mat<state_count_, 2 *L + 1> Xa = Xaa.topRows<state_count_>();
  const Mat<proc_noise_count_, 2 *L + 1> Wa =
      Xaa.bottomRows<proc_noise_count_>();

  // Apply process model
  for(unsigned int k = 0; k <= 2 * L; k++)
    Xa.col(k) = ProcessModel(Xa.col(k), u, Wa.col(k), dt);

  // Mean of sigma points
  const StateVec dx =
      wm0_ * Xa.col(0) + wm1_ * Xa.rightCols<2 * L>().rowwise().sum();

  state_ = state_.boxPlus(dx);

  // Covariance
  Pa_.setZero();
  for(unsigned int k = 1; k <= L; k++)
  {
    const StateVec d1 = Xa.col(k) - Xa.col(L + k);
    const StateVec d2 = Xa.col(k) + Xa.col(L + k) - 2 * Xa.col(0);
    Pa_.noalias() += wc1_ * d1 * d1.transpose() + wc2_ * d2 * d2.transpose();
  }
  return true;
}

bool AttitudeCDKF::MeasurementUpdateAccel(const MeasAccelVec &z,
                                          const MeasAccelCov &RnAccel,
                                          const ros::Time &time)
{
  // Init
  if(!init_meas_accel_)
  {
    ROS_INFO_STREAM("z.norm(): " <<  z.norm() << ", g_: " << g_);
    if(std::abs(z.norm() - g_) < 0.1)
    {
      const Vec<3> g_body = z.normalized();
      const Scalar_t pitch = -std::asin(g_body(0));
      const Scalar_t roll = std::atan2(g_body(1), g_body(2));
      ROS_INFO("Initial orientation: roll: %f, pitch: %f", 180 * roll / M_PI,
               180 * pitch / M_PI);
      const Scalar_t cp = std::cos(pitch / 2), sp = std::sin(pitch / 2);
      const Scalar_t cr = std::cos(roll / 2), sr = std::sin(roll / 2);
      Quat q;
      q.w() = cr * cp;
      q.x() = sr * cp;
      q.y() = cr * sp;
      q.z() = -sr * sp;
      state_.setOrientation(q);
      init_meas_accel_ = true;
    }
    return false;
  }

  constexpr unsigned int L = state_count_;

  // Generate sigma points
  const Mat<L, 2 *L + 1> Xaa = GenerateSigmaPoints(Mat<0,0>::Zero());
  const Mat<state_count_, 2 *L + 1> Xa = Xaa.topRows<state_count_>();

  // Apply measurement model
  Mat<meas_accel_count_, 2 * L + 1> Za;
  for(unsigned int k = 0; k <= 2 * L; k++)
    Za.col(k) = MeasurementModelAccel(Xa.col(k));

  // Mean of sigma points
  MeasAccelVec z_pred =
      wm0_ * Za.col(0) + wm1_ * Za.rightCols<2 * L>().rowwise().sum();

  // Covariance
  MeasAccelCov Pzz;
  Mat<state_count_, meas_accel_count_> Pxz;
  Pzz.setZero();
  Pxz.setZero();
  for(unsigned int k = 1; k <= L; k++)
  {
    const StateVec dx = Xa.col(k) - Xa.col(0);
    const MeasAccelVec dz1 = Za.col(k) - Za.col(L + k);
    const MeasAccelVec dz2 = Za.col(k) + Za.col(L + k) - 2 * Za.col(0);
    Pzz.noalias() +=
        wc1_ * dz1 * dz1.transpose() + wc2_ * dz2 * dz2.transpose();
    Pxz.noalias() += wm1_ * dx * dz1.transpose();
  }
  // Reduce weight of measurements when translational acceleration is high
  const Scalar_t diff_from_g_sq = (z.norm() - g_)*(z.norm() - g_);
  Pzz += (1 + 10 * diff_from_g_sq) * RnAccel;

  // Kalman Gain;
  const Mat<state_count_, meas_accel_count_> K = Pxz * Pzz.inverse();
  // Innovation
  const MeasAccelVec inno = z - z_pred;

  // Posterior Mean
  const StateVec dx = K * inno;
  state_ = state_.boxPlus(dx);

  // Posterior Covariance
  Pa_ -= K * Pzz * K.transpose();

  return true;
}

void AttitudeCDKF::GenerateWeights(unsigned int L)
{
  const Scalar_t h_sq = h_ * h_;
  wm0_ = (h_sq - L) / h_sq;
  wm1_ = 1 / (2 * h_sq);
  wc1_ = 1 / (4 * h_sq);
  wc2_ = (h_sq - 1) / (4 * h_sq * h_sq);
}

inline AttitudeCDKF::StateVec AttitudeCDKF::ProcessModel(const StateVec &dx,
                                                         const InputVec &u,
                                                         const ProcNoiseVec &w,
                                                         double dt)
{
  const State_t state_new = state_.boxPlus(dx);
  const Vec<3> ang_vel = u.topRows<3>() - state_new.getBias() + w.segment<3>(0);

  const Scalar_t d_theta = ang_vel.norm() * dt;
  const Vec<3> ang_vel_dir = ang_vel / ang_vel.norm();
  const Vec<3> ang_vel_vec = State_t::angle_axis_to_vec(d_theta, ang_vel_dir);

  StateVec dy;
  dy.head<3>() = ang_vel_vec;
  dy.tail<3>() = w.segment<3>(3) * dt;

  StateVec x_new = state_new.boxPlus(dy).boxMinus(state_);

  return x_new;
}

inline AttitudeCDKF::MeasAccelVec AttitudeCDKF::MeasurementModelAccel(
    const StateVec &dx)
{
  const Quat q_wb = state_.boxPlus(dx).getOrientation();
  const Vec<3> g_world{0, 0, g_};
  return q_wb.conjugate() * g_world;
}

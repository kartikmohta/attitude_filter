#ifndef ATTITUDE_STATE_H
#define ATTITUDE_STATE_H

#include <Eigen/Geometry>

template <typename Scalar_t>
class AttitudeState
{
 public:
  template <int N>
  using Vec = Eigen::Matrix<Scalar_t, N, 1>;

  template <int N, int M>
  using Mat = Eigen::Matrix<Scalar_t, N, M>;

  using Quat = Eigen::Quaternion<Scalar_t>;

  static constexpr int state_count_ = 6;

  using StateVec = Vec<state_count_>;
  using StateCov = Mat<state_count_, state_count_>;

  AttitudeState(const Quat &q = Quat::Identity(),
                const Vec<3> &bias = Vec<3>::Zero())
  {
    setOrientation(q);
    setBias(bias);
  }

  Quat getOrientation() const { return orientation_; }
  void setOrientation(const Quat &q) { orientation_ = q; }

  Vec<3> getBias() const { return gyro_bias_; }
  void setBias(const Vec<3> &b) { gyro_bias_ = b; }

  // x, y on the manifold, v in R^n
  // We want y = x.boxPlus(v) and v = y.boxMinus(x)
  AttitudeState<Scalar_t> boxPlus(const Vec<state_count_> &v) const
  {
    const Quat q = orientation_ * vec_to_quat(v.template head<3>());
    const Vec<3> bias = gyro_bias_ + v.template tail<3>();
    return AttitudeState<Scalar_t>(q, bias);
  }

  StateVec boxMinus(const AttitudeState<Scalar_t> &s) const
  {
    StateVec v;
    v.template head<3>() =
        quat_to_vec(s.getOrientation().conjugate() * orientation_);
    v.template tail<3>() = gyro_bias_ - s.getBias();
    return v;
  }

  static Vec<3> angle_axis_to_vec(const Scalar_t &angle, const Vec<3> &axis)
  {
    return 4 * axis * std::tan(angle / 4);
  }

 private:

  Vec<3> quat_to_vec(const Quat &q) const
  {
    return 4 * q.vec() / (1 + q.w());
  }

  Quat vec_to_quat(const Vec<3> &vec) const
  {
    Quat q;
    q.w() = (16 - vec.squaredNorm()) / (16 + vec.squaredNorm());
    q.vec() = (1 + q.w()) * vec / 4;
    return q;
  }

  Quat orientation_;
  Vec<3> gyro_bias_;
};

#endif

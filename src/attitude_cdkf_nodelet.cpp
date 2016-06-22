#include "attitude_filter/attitude_cdkf.h"
#include <Eigen/Geometry>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>

using Scalar_t = AttitudeCDKF::Scalar_t;

template <int N>
using Vec = AttitudeCDKF::Vec<N>;

class AttitudeCDKFNodelet : public nodelet::Nodelet
{
 public:
  AttitudeCDKFNodelet();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // Need this since we have AttitudeCDKF which
                                   // needs aligned pointer

 private:
  void onInit(void);
  void imu_callback(const sensor_msgs::Imu::ConstPtr &msg);

  AttitudeCDKF attitude_cdkf_;
  ros::Subscriber sub_imu_;
  ros::Publisher pub_pose_;
  std::string frame_id_;
  std::string child_frame_id_;
  static const int imu_calib_limit_ = 200;
  int imu_calib_count_;
  Vec<3> acc_gravity_;
  AttitudeCDKF::MeasAccelCov RnAccel_;
};

AttitudeCDKFNodelet::AttitudeCDKFNodelet()
    : imu_calib_count_(0), acc_gravity_(Vec<3>::Zero())
{
}

void AttitudeCDKFNodelet::imu_callback(const sensor_msgs::Imu::ConstPtr &msg)
{
  // Assemble control input
  AttitudeCDKF::InputVec u;
  u(0) = msg->angular_velocity.x;
  u(1) = msg->angular_velocity.y;
  u(2) = msg->angular_velocity.z;

  const AttitudeCDKF::MeasAccelVec acc{msg->linear_acceleration.x,
                                       msg->linear_acceleration.y,
                                       msg->linear_acceleration.z};
  if(imu_calib_count_ < imu_calib_limit_) // Calibration
  {
    acc_gravity_ += acc;
    imu_calib_count_++;
  }
  else if(imu_calib_count_ == imu_calib_limit_) // Save gravity norm
  {
    imu_calib_count_++;
    acc_gravity_ /= imu_calib_limit_;
    const Scalar_t g = acc_gravity_.norm();
    ROS_INFO("Setting gravity = %f", g);
    attitude_cdkf_.SetGravity(g);
  }
  else
  {
    // Process Update
    bool publish = attitude_cdkf_.ProcessUpdate(u, msg->header.stamp);

    // Measurement update
    attitude_cdkf_.MeasurementUpdateAccel(acc, RnAccel_, msg->header.stamp);

    if(publish)
    {
      geometry_msgs::PoseWithCovarianceStamped pose_out;
      pose_out.header.stamp = msg->header.stamp;
      pose_out.header.frame_id = frame_id_;
      const AttitudeCDKF::State_t state = attitude_cdkf_.GetState();
      const auto q = state.getOrientation();
      pose_out.pose.pose.orientation.x = q.x();
      pose_out.pose.pose.orientation.y = q.y();
      pose_out.pose.pose.orientation.z = q.z();
      pose_out.pose.pose.orientation.w = q.w();
      // Bias
      const auto bias = state.getBias();
      pose_out.pose.pose.position.x = bias(0);
      pose_out.pose.pose.position.y = bias(1);
      pose_out.pose.pose.position.z = bias(2);
      AttitudeCDKF::StateCov P = attitude_cdkf_.GetStateCovariance();
      for(int j = 0; j < 6; j++)
        for(int i = 0; i < 6; i++)
          pose_out.pose.covariance[6 * j + i] =
              P((i < 3) ? i + 3 : i - 3, (j < 3) ? j + 3 : j - 3);
      // Publish Msg
      pub_pose_.publish(pose_out);
    }
  }
}

void AttitudeCDKFNodelet::onInit(void)
{
  ros::NodeHandle n(getPrivateNodeHandle());

  // CDKF Parameters
  double h;

  n.param("frame_id", frame_id_, std::string("/world"));
  n.param("child_frame_id", child_frame_id_, std::string("robot"));

  n.param("h", h, std::sqrt(3));

  // Noise standard devs
  double std_accel_noise[3], std_gyro_noise[3], std_gyro_bias[3];
  n.param("noise_std/process/w/x", std_gyro_noise[0], 0.1);
  n.param("noise_std/process/w/y", std_gyro_noise[1], 0.1);
  n.param("noise_std/process/w/z", std_gyro_noise[2], 0.1);
  n.param("noise_std/process/gyro_bias/x", std_gyro_bias[0], 0.0001);
  n.param("noise_std/process/gyro_bias/y", std_gyro_bias[1], 0.0001);
  n.param("noise_std/process/gyro_bias/z", std_gyro_bias[2], 0.0001);
  n.param("noise_std/meas/acc/x", std_accel_noise[0], 0.1);
  n.param("noise_std/meas/acc/y", std_accel_noise[1], 0.1);
  n.param("noise_std/meas/acc/z", std_accel_noise[2], 0.1);

  // Fixed process noise
  AttitudeCDKF::ProcNoiseCov Rv;
  Rv.setZero();
  Rv(0, 0) = std_gyro_noise[0] * std_gyro_noise[0];
  Rv(1, 1) = std_gyro_noise[1] * std_gyro_noise[1];
  Rv(2, 2) = std_gyro_noise[2] * std_gyro_noise[2];
  Rv(3, 3) = std_gyro_bias[0] * std_gyro_bias[0];
  Rv(4, 4) = std_gyro_bias[1] * std_gyro_bias[1];
  Rv(5, 5) = std_gyro_bias[2] * std_gyro_bias[2];

  // Fixed measurement noise
  RnAccel_.setZero();
  RnAccel_(0, 0) = std_accel_noise[0] * std_accel_noise[0];
  RnAccel_(1, 1) = std_accel_noise[1] * std_accel_noise[1];
  RnAccel_(2, 2) = std_accel_noise[2] * std_accel_noise[2];

  // Initialize CDKF
  attitude_cdkf_.SetParameter(h);
  attitude_cdkf_.SetImuCovariance(Rv);

  pub_pose_ =
      n.advertise<geometry_msgs::PoseWithCovarianceStamped>("pose_out", 10);
  sub_imu_ = n.subscribe("imu", 10, &AttitudeCDKFNodelet::imu_callback, this,
                         ros::TransportHints().tcpNoDelay());
}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(AttitudeCDKFNodelet, nodelet::Nodelet);

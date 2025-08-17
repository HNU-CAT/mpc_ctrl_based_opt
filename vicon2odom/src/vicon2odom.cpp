#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <boost/bind.hpp>
#include <Eigen/Dense>
#include <tf2/LinearMath/Quaternion.h>

// 定义常用类型简化代码
using V3D = Eigen::Vector3d;
using Quat = Eigen::Quaterniond;

// 四元数指数映射（旋转向量到四元数）
Quat Exp(const V3D& omega, double dt) {
    V3D omega_dt = omega * dt;
    double theta = omega_dt.norm();
    if (theta < 1e-8) {
        return Quat(1, omega_dt.x()/2, omega_dt.y()/2, omega_dt.z()/2);
    } else {
        double sin_half = sin(theta/2);
        double cos_half = cos(theta/2);
        return Quat(cos_half, 
                   omega_dt.x()/theta * sin_half, 
                   omega_dt.y()/theta * sin_half, 
                   omega_dt.z()/theta * sin_half);
    }
}

class OdomPublisher {
public:
    OdomPublisher(ros::NodeHandle& nh) : nh_(nh), is_initialized_(false), 
                                        last_mocap_time_(0), gravity_(9.81),
                                        imu_calib_count_(0), imu_calib_samples_(100) {
        // 加载参数（含IMU零偏配置）
        loadParameters();
        
        // 订阅动捕pose和twist（带时间同步）
        pose_sub_.subscribe(nh_, pose_topic_, 10);
        twist_sub_.subscribe(nh_, twist_topic_, 10);
        sync_.reset(new Sync(MySyncPolicy(20), pose_sub_, twist_sub_));
        sync_->registerCallback(boost::bind(&OdomPublisher::mocapCallback, this, _1, _2));
        
        // 根据参数决定是否订阅IMU
        if (use_imu_interpolation_) {
            imu_sub_ = nh_.subscribe(imu_topic_, 100, &OdomPublisher::imuCallback, this);
            ROS_INFO("IMU interpolation enabled, subscribing to IMU topic: %s", imu_topic_.c_str());
        } else {
            ROS_INFO("IMU interpolation disabled, using raw mocap data");
        }
        
        // 初始化发布器
        odom_pub_ = nh_.advertise<nav_msgs::Odometry>(odom_topic_, 100);
        pubOdomtoPx4 = nh_.advertise<geometry_msgs::PoseStamped>(px4_pose_topic_, 100);
        
        // 初始化状态变量（含零偏）
        resetState();
        logParameters();
    }

private:
    ros::NodeHandle nh_;
    
    // 参数配置（新增IMU零偏参数）
    std::string pose_topic_;      // 动捕pose话题
    std::string twist_topic_;     // 动捕twist话题
    std::string imu_topic_;       // 飞控IMU话题
    std::string odom_topic_;      // 输出odom话题
    std::string px4_pose_topic_;  // PX4 pose话题
    std::string frame_id_;        // 世界坐标系
    std::string child_frame_id_;  // 机体坐标系
    bool use_imu_interpolation_;  // 是否启用IMU插值
    bool use_imu_filter_;         // 是否对IMU数据滤波
    bool auto_calib_bias_;        // 是否自动校准IMU零偏
    double gravity_;              // 重力加速度（m/s²）
    V3D init_Ba_;                 // 初始加速度计零偏
    V3D init_Bg_;                 // 初始陀螺仪零偏

    // 动捕数据同步器
    message_filters::Subscriber<geometry_msgs::PoseStamped> pose_sub_;
    message_filters::Subscriber<geometry_msgs::TwistStamped> twist_sub_;
    typedef message_filters::sync_policies::ApproximateTime<
        geometry_msgs::PoseStamped, geometry_msgs::TwistStamped
    > MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    boost::shared_ptr<Sync> sync_;

    // 订阅器和发布器
    ros::Subscriber imu_sub_;
    ros::Publisher odom_pub_;
    ros::Publisher pubOdomtoPx4;

    // 状态变量（含零偏）
    struct State {
        double time;              // 时间戳
        V3D pos;                  // 位置
        Quat rot;                 // 姿态（四元数）
        V3D vel;                  // 速度
        V3D acc;                  // 加速度（机体系）
        V3D gyr;                  // 角速度（机体系）
        V3D Ba;                   // 加速度计零偏
        V3D Bg;                   // 陀螺仪零偏
    } latest_state_, predicted_state_;

    // 辅助变量（用于零偏校准）
    bool is_initialized_;         // 是否初始化完成
    double last_mocap_time_;      // 上一次动捕数据时间
    V3D acc_raw_[2];              // 加速度原始数据缓存（用于滤波）
    V3D acc_filtered_;            // 滤波后的加速度
    V3D gyr_raw_[2];              // 角速度原始数据缓存（用于滤波）
    V3D gyr_filtered_;            // 滤波后的角速度
    int imu_calib_count_;         // 校准样本计数
    int imu_calib_samples_;       // 校准所需样本数
    V3D acc_calib_sum_;           // 加速度校准总和
    V3D gyr_calib_sum_;           // 角速度校准总和


    // 加载参数（新增IMU零偏配置）
    void loadParameters() {
        nh_.param<std::string>("pose_topic", pose_topic_, "/vrpn_client_node/uav1/pose");
        nh_.param<std::string>("twist_topic", twist_topic_, "/vrpn_client_node/uav1/twist");
        nh_.param<std::string>("imu_topic", imu_topic_, "/mavros/imu/data");
        nh_.param<std::string>("odom_topic", odom_topic_, "/vision_odom");
        nh_.param<std::string>("px4_pose_topic", px4_pose_topic_, "/mavros/vision_pose/pose");
        nh_.param<std::string>("frame_id", frame_id_, "odom");
        nh_.param<std::string>("child_frame_id", child_frame_id_, "base_link");
        nh_.param<bool>("use_imu_interpolation", use_imu_interpolation_, true);
        nh_.param<bool>("use_imu_filter", use_imu_filter_, true);
        nh_.param<bool>("auto_calib_bias", auto_calib_bias_, true);  // 自动校准零偏开关
        nh_.param<double>("gravity", gravity_, 9.81);

        // 加载初始零偏参数（从launch文件配置）
        double Ba_x, Ba_y, Ba_z;
        double Bg_x, Bg_y, Bg_z;
        nh_.param<double>("init_Ba_x", Ba_x, 0.0);
        nh_.param<double>("init_Ba_y", Ba_y, 0.0);
        nh_.param<double>("init_Ba_z", Ba_z, 0.0);
        init_Ba_ = V3D(Ba_x, Ba_y, Ba_z);

        nh_.param<double>("init_Bg_x", Bg_x, 0.0);
        nh_.param<double>("init_Bg_y", Bg_y, 0.0);
        nh_.param<double>("init_Bg_z", Bg_z, 0.0);
        init_Bg_ = V3D(Bg_x, Bg_y, Bg_z);
    }

    // 日志输出参数（含零偏信息）
    void logParameters() {
        ROS_INFO("Odom Publisher initialized with parameters:");
        ROS_INFO("  Mocap pose topic: %s", pose_topic_.c_str());
        ROS_INFO("  Mocap twist topic: %s", twist_topic_.c_str());
        ROS_INFO("  IMU interpolation enabled: %s", use_imu_interpolation_ ? "true" : "false");
        if (use_imu_interpolation_) {
            ROS_INFO("  IMU input topic: %s", imu_topic_.c_str());
            ROS_INFO("  IMU filter enabled: %s", use_imu_filter_ ? "true" : "false");
            ROS_INFO("  Auto-calibrate bias: %s", auto_calib_bias_ ? "true" : "false");
            ROS_INFO("  Initial accelerometer bias (Ba): [%.4f, %.4f, %.4f]", 
                     init_Ba_.x(), init_Ba_.y(), init_Ba_.z());
            ROS_INFO("  Initial gyroscope bias (Bg): [%.4f, %.4f, %.4f]", 
                     init_Bg_.x(), init_Bg_.y(), init_Bg_.z());
        }
        ROS_INFO("  Output odom topic: %s", odom_topic_.c_str());
        ROS_INFO("  Output PX4 pose topic: %s", px4_pose_topic_.c_str());
    }

    // 重置状态变量（初始化零偏）
    void resetState() {
        latest_state_.time = 0;
        latest_state_.pos.setZero();
        latest_state_.rot.setIdentity();
        latest_state_.vel.setZero();
        latest_state_.acc.setZero();
        latest_state_.gyr.setZero();
        latest_state_.Ba = init_Ba_;  // 初始化加速度计零偏
        latest_state_.Bg = init_Bg_;  // 初始化陀螺仪零偏

        predicted_state_ = latest_state_;
        acc_raw_[0].setZero();
        acc_raw_[1].setZero();
        acc_filtered_.setZero();
        gyr_raw_[0].setZero();
        gyr_raw_[1].setZero();
        gyr_filtered_.setZero();

        // 校准相关变量重置
        imu_calib_count_ = 0;
        acc_calib_sum_.setZero();
        gyr_calib_sum_.setZero();
    }

    // 动捕数据回调（更新最新状态）
    void mocapCallback(const geometry_msgs::PoseStamped::ConstPtr& pose_msg,
                      const geometry_msgs::TwistStamped::ConstPtr& twist_msg) {
        // 转换动捕数据到状态变量
        latest_state_.time = pose_msg->header.stamp.toSec();
        last_mocap_time_ = latest_state_.time;

        // 位置
        latest_state_.pos.x() = pose_msg->pose.position.x;
        latest_state_.pos.y() = pose_msg->pose.position.y;
        latest_state_.pos.z() = pose_msg->pose.position.z;

        // 姿态（四元数）
        latest_state_.rot.w() = pose_msg->pose.orientation.w;
        latest_state_.rot.x() = pose_msg->pose.orientation.x;
        latest_state_.rot.y() = pose_msg->pose.orientation.y;
        latest_state_.rot.z() = pose_msg->pose.orientation.z;
        latest_state_.rot.normalize();

        // 速度
        latest_state_.vel.x() = twist_msg->twist.linear.x;
        latest_state_.vel.y() = twist_msg->twist.linear.y;
        latest_state_.vel.z() = twist_msg->twist.linear.z;

        // 初始化标记
        if (!is_initialized_) {
            is_initialized_ = true;
            predicted_state_ = latest_state_;
            ROS_INFO("Mocap data received, initialization complete");
        }

        // 若不启用IMU插值，直接发布动捕数据
        if (!use_imu_interpolation_) {
            publishState(latest_state_);
        } else {
            // 启用IMU插值时，用动捕数据校准预测状态（包括零偏）
            predicted_state_ = latest_state_;
        }
    }

    // IMU数据回调（含零偏校准）
    void imuCallback(const sensor_msgs::Imu::ConstPtr& imu_msg) {
        if (!use_imu_interpolation_ || !is_initialized_) return;

        double imu_time = imu_msg->header.stamp.toSec();
        double dt = imu_time - predicted_state_.time;
        if (dt <= 0 || dt > 0.1) {
            predicted_state_.time = imu_time;
            return;
        }

        // 读取IMU原始数据
        V3D acc, gyr;
        acc << imu_msg->linear_acceleration.x, 
               imu_msg->linear_acceleration.y, 
               imu_msg->linear_acceleration.z;
        gyr << imu_msg->angular_velocity.x, 
               imu_msg->angular_velocity.y, 
               imu_msg->angular_velocity.z;

        // 自动校准零偏（系统启动时，IMU静止状态下采集样本）
        if (auto_calib_bias_ && imu_calib_count_ < imu_calib_samples_) {
            acc_calib_sum_ += acc;
            gyr_calib_sum_ += gyr;
            imu_calib_count_++;
            
            // 校准完成，更新零偏
            if (imu_calib_count_ == imu_calib_samples_) {
                predicted_state_.Ba = acc_calib_sum_ / imu_calib_samples_ - V3D(0, 0, gravity_);
                predicted_state_.Bg = gyr_calib_sum_ / imu_calib_samples_;
                latest_state_.Ba = predicted_state_.Ba;
                latest_state_.Bg = predicted_state_.Bg;
                
                ROS_INFO("IMU bias calibration complete:");
                ROS_INFO("  Calculated Ba: [%.4f, %.4f, %.4f]", 
                         predicted_state_.Ba.x(), predicted_state_.Ba.y(), predicted_state_.Ba.z());
                ROS_INFO("  Calculated Bg: [%.4f, %.4f, %.4f]", 
                         predicted_state_.Bg.x(), predicted_state_.Bg.y(), predicted_state_.Bg.z());
            }
            return;  // 校准期间不进行状态预测
        }

        // IMU数据低通滤波
        if (use_imu_filter_) {
            acc_raw_[0] = acc_raw_[1];
            acc_raw_[1] = acc;
            acc_filtered_ = 0.2 * acc_raw_[0] + 0.3 * acc_raw_[1] + 0.5 * acc_filtered_;
            acc = acc_filtered_;

            gyr_raw_[0] = gyr_raw_[1];
            gyr_raw_[1] = gyr;
            gyr_filtered_ = 0.2 * gyr_raw_[0] + 0.3 * gyr_raw_[1] + 0.5 * gyr_filtered_;
            gyr = gyr_filtered_;
        }

        // 使用IMU数据预测状态（含零偏补偿）
        predictState(imu_time, dt, acc, gyr);

        // 发布预测状态
        publishState(predicted_state_);
    }

    // 基于IMU数据预测状态（零偏补偿版）
    void predictState(double time, double dt, const V3D& acc, const V3D& gyr) {
        // 1. 姿态预测（扣除陀螺仪零偏）
        V3D un_gyr = gyr - predicted_state_.Bg;  // 关键：减去陀螺仪零偏
        predicted_state_.rot = predicted_state_.rot * Exp(un_gyr, dt);
        predicted_state_.rot.normalize();

        // 2. 加速度转换到世界系（扣除加速度计零偏+重力）
        V3D un_acc = predicted_state_.rot * (acc - predicted_state_.Ba) + V3D(0, 0, gravity_);  // 关键：减去加速度计零偏

        // 3. 位置和速度预测
        predicted_state_.pos += predicted_state_.vel * dt + 0.5 * un_acc * dt * dt;
        predicted_state_.vel += un_acc * dt;

        // 4. 更新时间和原始IMU数据
        predicted_state_.time = time;
        predicted_state_.acc = acc;
        predicted_state_.gyr = gyr;
    }

    // 发布状态数据
    void publishState(const State& state) {
        // 1. 发布odom话题
        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp = ros::Time(state.time);
        odom_msg.header.frame_id = frame_id_;
        odom_msg.child_frame_id = child_frame_id_;

        // 位置和姿态
        odom_msg.pose.pose.position.x = state.pos.x();
        odom_msg.pose.pose.position.y = state.pos.y();
        odom_msg.pose.pose.position.z = state.pos.z();
        odom_msg.pose.pose.orientation.w = state.rot.w();
        odom_msg.pose.pose.orientation.x = state.rot.x();
        odom_msg.pose.pose.orientation.y = state.rot.y();
        odom_msg.pose.pose.orientation.z = state.rot.z();

        // 速度
        odom_msg.twist.twist.linear.x = state.vel.x();
        odom_msg.twist.twist.linear.y = state.vel.y();
        odom_msg.twist.twist.linear.z = state.vel.z();
        odom_msg.twist.twist.angular.x = state.gyr.x();
        odom_msg.twist.twist.angular.y = state.gyr.y();
        odom_msg.twist.twist.angular.z = state.gyr.z();

        // 协方差矩阵
        odom_msg.pose.covariance = {
            0.01, 0, 0, 0, 0, 0,
            0, 0.01, 0, 0, 0, 0,
            0, 0, 0.01, 0, 0, 0,
            0, 0, 0, 0.02, 0, 0,
            0, 0, 0, 0, 0.02, 0,
            0, 0, 0, 0, 0, 0.02
        };
        odom_msg.twist.covariance = {
            0.01, 0, 0, 0, 0, 0,
            0, 0.01, 0, 0, 0, 0,
            0, 0, 0.01, 0, 0, 0,
            0, 0, 0, 0.02, 0, 0,
            0, 0, 0, 0, 0.02, 0,
            0, 0, 0, 0, 0, 0.02
        };
        odom_pub_.publish(odom_msg);

        // 2. 发布给PX4的pose话题
        geometry_msgs::PoseStamped pose_msg_px4;
        pose_msg_px4.header.stamp = odom_msg.header.stamp;
        pose_msg_px4.header.frame_id = frame_id_;
        pose_msg_px4.pose = odom_msg.pose.pose;
        pubOdomtoPx4.publish(pose_msg_px4);
    }
};

int main(int argc, char**argv) {
    ros::init(argc, argv, "vicon2odom_node");
    ros::NodeHandle nh("~");  // 使用私有命名空间读取参数
    OdomPublisher odom_publisher(nh);
    ros::spin();
    return 0;
}
    
#include "controller.h"
#include "Marginalization.h"

using namespace gtsam;
using namespace std;
using namespace dmvio;

using namespace uavfactor;

using symbol_shorthand::B;
using symbol_shorthand::R;
using symbol_shorthand::S;
using symbol_shorthand::U;
using symbol_shorthand::V;
using symbol_shorthand::X;

static std::random_device __randomDevice;
static std::mt19937 __randomGen(__randomDevice());

bool Controller::initializeState(const std::vector<Imu_Data_t> &imu_raw, const std::vector<Odom_Data_t> &fakeGPS,
                                 gtsam::Vector3 &init_vel, gtsam::Vector6 &init_bias)
{
  double opt_cost = 0.0f;
  clock_t start, end;

  gtsam_fg graph_init;
  gtsam_sols initial_value;

  // IMU noise
  auto imu_factor_noise = noiseModel::Diagonal::Sigmas((Vector(9) << Vector3::Constant(param_.factor_graph.acc_sigma_x * dt_ * dt_ * 0.5f + param_.factor_graph.acc_sigma_x * dt_ * dt_),
                                                        Vector3::Constant(param_.factor_graph.gyro_sigma_x * dt_), Vector3::Constant(param_.factor_graph.acc_sigma_x * dt_))
                                                           .finished());

  // Bias noise
  auto bias_noise = noiseModel::Diagonal::Sigmas(
      (Vector(6) << Vector3(param_.factor_graph.acc_bias_imu_x, param_.factor_graph.acc_bias_imu_x, param_.factor_graph.acc_bias_imu_x),
       Vector3(param_.factor_graph.gyro_bias_sigma_x, param_.factor_graph.gyro_bias_sigma_x, param_.factor_graph.gyro_bias_sigma_x))
          .finished());

  // Prior noise
  auto prior_bias_noise = noiseModel::Diagonal::Sigmas(
      (Vector(6) << Vector3::Constant(param_.factor_graph.prior_acc_sigma), Vector3::Constant(param_.factor_graph.prior_gyro_sigma)).finished());
  auto prior_vicon_noise = noiseModel::Diagonal::Sigmas(
      (Vector(6) << Vector3::Constant(param_.factor_graph.PRIOR_ROT_MEAS_COV), Vector3::Constant(param_.factor_graph.PRIOR_POS_MEAS_COV)).finished());
  auto prior_vel_noise = noiseModel::Diagonal::Sigmas(
      Vector3(param_.factor_graph.PRIOR_VEL_MEAS_COV, param_.factor_graph.PRIOR_VEL_MEAS_COV, param_.factor_graph.PRIOR_VEL_MEAS_COV));

  // GPS noise
  auto noise_model_gps = noiseModel::Isotropic::Sigma(3, param_.factor_graph.POS_MEAS_COV);
  // gtsam::GPSFactor gps_factor(X(correction_count), Point3(gps(0), gps(1), gps(2)), noise_model_gps);

  for (uint16_t idx = 0; idx < window_lens_; idx++)
  {
    gtsam::Pose3 pose = gtsam::Pose3(gtsam::Rot3(fakeGPS[idx].q), fakeGPS[idx].p);
    gtsam::Vector3 vel = fakeGPS[idx].v;

    if (idx != 0)
    {
      float __dt = (fakeGPS[idx].rcv_stamp - fakeGPS[idx - 1].rcv_stamp).toSec();
      graph_init.add(IMUFactor(X(idx - 1), V(idx - 1), B(idx - 1), X(idx), V(idx), __dt, imu_raw[idx].a, imu_raw[idx].w, imu_factor_noise));
      gtsam_imuBi zero_bias(gtsam::Vector3(0, 0, 0), gtsam::Vector3(0, 0, 0));
      graph_init.add(BetweenFactor<gtsam_imuBi>(B(idx - 1), B(idx), zero_bias, bias_noise));
    }

    // graph_init.add(gtsam::GPSFactor(X(idx), odom_v[idx].p, noise_model_gps));
    graph_init.add(gtsam::PriorFactor<gtsam::Pose3>(X(idx), pose, prior_vicon_noise));
    // graph_init.add(gtsam::PriorFactor<gtsam::Vector3>(V(idx), vel,  prior_vel_noise));

    initial_value.insert(B(idx), gtsam_imuBi());
    initial_value.insert(X(idx), pose);
    initial_value.insert(V(idx), vel);
  }

  gtsam::LevenbergMarquardtParams parameters;
  parameters.absoluteErrorTol = 100;
  parameters.relativeErrorTol = 1e-2;
  parameters.maxIterations = 10;
  parameters.verbosity = gtsam::NonlinearOptimizerParams::SILENT;
  parameters.verbosityLM = gtsam::LevenbergMarquardtParams::SILENT;

  std::cout << " -- <  Initialization Test > -- " << std::endl;
  LevenbergMarquardtOptimizer optimizer(graph_init, initial_value, parameters);
  start = clock();
  Values result = optimizer.optimize();
  end = clock();
  std::cout << " ---------------------------------------------------- Result ----------------------------------------------------" << std::endl;
  // result.print();
  opt_cost = (double)(end - start) / CLOCKS_PER_SEC;
  std::cout << " ---------- Initialization Time: [ " << opt_cost << " ] " << endl;

  gtsam::Vector3 vel;
  gtsam_imuBi imu_bias;
  imu_bias = result.at<gtsam_imuBi>(B(0));
  init_vel = result.at<Vector3>(V(0));
  init_bias = imu_bias.vector();

  std::cout << "Initialization Vel:  [ " << init_vel.transpose() << " ] " << endl;
  std::cout << "Initialization Bias: [ " << init_bias.transpose() << " ] " << endl;

  return true;
}

/* Fusion */
quadrotor_msgs::Px4ctrlDebug Controller::fusion(const Odom_Data_t &odom, const Imu_Data_t &imu_raw, const Odom_Data_t &GT, gtsam::Pose3 &fus_pose, gtsam::Vector3 &fus_vel, gtsam::Vector3 &fus_w)
{
  odom_data_v_.push_back(GT);
  gtsam::Vector3 gt_rxyz = gtsam::Rot3(GT.q).rpy();

  odom_data_noise_.push_back(odom);
  imu_data_v_.push_back(imu_raw);

  double opt_cost = 0.0f;

  clock_t start, end;
  gtsam::Vector3 init_vel;
  gtsam::Vector6 init_bias;

  if (odom_data_v_.size() == window_lens_)
  {
    if (!init_state_flag_)
    {
      initializeState(imu_data_v_, odom_data_v_, init_vel_, init_bias_);
      init_state_flag_ = true;
    }

    gtsam::LevenbergMarquardtParams parameters;
    parameters.absoluteErrorTol = 100;
    parameters.relativeErrorTol = 1e-2;
    parameters.maxIterations = 10;
    parameters.verbosity = gtsam::NonlinearOptimizerParams::SILENT;
    parameters.verbosityLM = gtsam::LevenbergMarquardtParams::SILENT;

    FGbuilder->buildFusionFG(graph_, initial_value_, odom_data_noise_, imu_data_v_, dt_, state_idx_);
    LevenbergMarquardtOptimizer optimizer(graph_, initial_value_, parameters);
    start = clock();
    Values result = optimizer.optimize();
    end = clock();
    initial_value_ = result;
    std::cout << " ---------------------------------------------------- Result ----------------------------------------------------" << std::endl;
    result.print();
    opt_cost = (double)(end - start) / CLOCKS_PER_SEC;
    std::cout << " ---------- Optimize Time: [ " << opt_cost << " ] " << endl;

    uint16_t idx = 0;
    if (state_idx_ == 1)
    {
      idx = window_lens_ - 1 + IDX_P_START;
    }
    else
    {
      idx = state_idx_ - 1 + IDX_P_START;
    }

    gtsam::Pose3 pose;
    gtsam::Vector3 vel;
    gtsam_imuBi imu_bias;
    gtsam::Rot3 Rg;

    pose = result.at<Pose3>(X(idx));
    vel = result.at<Vector3>(V(idx));
    imu_bias = result.at<gtsam_imuBi>(B(idx));

    fus_pose = pose;
    fus_vel = vel;
    fus_w = imu_bias.correctGyroscope(imu_data_v_[window_lens_ - 1].w); // Guassian noise

    if (param_.factor_graph.opt_gravity_rot)
    {
      Rg = result.at<gtsam::Rot3>(R(0));
      std::cout << " --- Gravity rotation:" << Rg.rpy().transpose() << std::endl;
    }
    gtsam::Vector3 fusion_rxyz = pose.rotation().rpy();

    log_ << std::setprecision(19)
         // GT
         << GT.p.x() << " " << GT.p.y() << " " << GT.p.z() << " "
         << gt_rxyz.x() << " " << gt_rxyz.y() << " " << gt_rxyz.z() << " "
         << GT.v.x() << " " << GT.v.y() << " " << GT.v.z() << " "

         // Measurement
         << odom.p.x() << " " << odom.p.y() << " " << odom.p.z() << " "
         << odom.v.x() << " " << odom.v.y() << " " << odom.v.z() << " "

         // Estimation
         << pose.translation().x() << " " << pose.translation().y() << " " << pose.translation().z() << " "
         << vel.x() << " " << vel.y() << " " << vel.z() << " "
         << fusion_rxyz.x() << " " << fusion_rxyz.y() << " " << fusion_rxyz.z() << " "
         // Time Cost
         << opt_cost << " "

         // IMU Raw Data
         << imu_raw.w.x() << " " << imu_raw.w.y() << " " << imu_raw.w.z() << " "
         << imu_raw.a.x() << " " << imu_raw.a.y() << " " << imu_raw.a.z() << " "

         // IMU Bias
         << imu_bias.accelerometer().x() << " " << imu_bias.accelerometer().y() << " " << imu_bias.accelerometer().z() << " "
         << imu_bias.gyroscope().x() << " " << imu_bias.gyroscope().y() << " " << imu_bias.gyroscope().z() << " "

         << std::endl;
  }

  if (odom_data_v_.size() >= window_lens_)
  {
    odom_data_v_.erase(odom_data_v_.begin());
  }

  if (odom_data_noise_.size() >= window_lens_)
  {
    odom_data_noise_.erase(odom_data_noise_.begin());
  }

  if (imu_data_v_.size() >= window_lens_)
  {
    imu_data_v_.erase(imu_data_v_.begin());
  }

  return debug_msg_;
}

/* JPCM */
quadrotor_msgs::Px4ctrlDebug Controller::calculateControl(const Desired_State_t &des, const Odom_Data_t &odom, const Imu_Data_t &imu,
                                                          const Imu_Data_t &imu_raw,
                                                          Controller_Output_t &thr_bodyrate_u, CTRL_MODE mode_switch)
{
  odom_data_v_.push_back(odom);
  Odom_Data_t odom_noise = add_Guassian_noise(odom);

  odom_data_noise_.push_back(odom_noise);
  des_data_v_.push_back(des);
  imu_data_v_.push_back(imu_raw);

  bool timeout = false;
  double opt_cost = 0.0f;

  double thrust2 = 0;
  gtsam::Vector3 bodyrates2(0, 0, 0);

  if (timeout || mode_switch == DFBC || des_data_v_.size() < opt_traj_lens_)
  {
    Controller::calculateControl(des_data_v_[0], odom_noise, imu, thr_bodyrate_u);
  }
  else if (mode_switch == JPCM && des_data_v_.size() == opt_traj_lens_) // && odom.p.z() > hight_thr)
  {
    clock_t start, end;
    // 优化参数配置
    gtsam::LevenbergMarquardtParams parameters;
    parameters.absoluteErrorTol = 100;
    parameters.relativeErrorTol = 1e-2;
    parameters.maxIterations = 10;
    parameters.verbosity = gtsam::NonlinearOptimizerParams::SILENT;
    parameters.verbosityLM = gtsam::LevenbergMarquardtParams::SILENT;

    std::cout << " - JPCM Opt - " << std::endl;
    // 调用因子图构建器FGbuilder的buildFactorGraph方法，构建因子图graph_并设置初始值initial_value_
    FGbuilder->buildFactorGraph(graph_, initial_value_, des_data_v_, odom_data_noise_, imu_data_v_, dt_, state_idx_);
    LevenbergMarquardtOptimizer optimizer(graph_, initial_value_, parameters);
    start = clock();
    Values result = optimizer.optimize();
    end = clock();
    initial_value_ = result;
    std::cout << " ---------------------------------------------------- Result ----------------------------------------------------" << std::endl;
    // result.print();
    opt_cost = (double)(end - start) / CLOCKS_PER_SEC;
    float distance = (des_data_v_[0].p - odom_noise.p).norm();

    gtsam::Vector4 input;
    // 从优化结果result中提取索引为U(0)的控制输入变量（U(0)表示当前时刻的控制输入，定义见前序因子图构建逻辑）
    input = result.at<gtsam::Vector4>(U(0));
    Eigen::Vector3d des_acc(0, 0, input[0]);
    // 调用computeDesiredCollectiveThrustSignal函数，将期望加速度转换为实际推力值thrust2（通常考虑质量和重力补偿）
    thrust2 = Controller::computeDesiredCollectiveThrustSignal(des_acc);
    bodyrates2 = Eigen::Vector3d(input[1], input[2], input[3]);
    // 将优化得到的推力和机体速率赋值给控制输出thr_bodyrate_u，完成控制量计算
    thr_bodyrate_u.thrust = thrust2;
    thr_bodyrate_u.bodyrates = bodyrates2;

    uint16_t idx = 0;
    if (state_idx_ == 1)
    {
      idx = window_lens_ - 1 + IDX_P_START;
    }
    else
    {
      idx = state_idx_ - 1 + IDX_P_START;
    }

    gtsam::Pose3 pose;
    gtsam::Vector3 vel;
    gtsam_imuBi imu_bias;

    pose = result.at<Pose3>(X(idx));
    vel = result.at<Vector3>(V(idx));
    imu_bias = result.at<gtsam_imuBi>(B(idx));

    gtsam::Vector3 eular_xyz = pose.rotation().rpy();
    gtsam::Vector3 gt_eular_xyz = gtsam::Rot3(odom.q).rpy();
    gtsam::Vector3 des_eular_xyz = gtsam::Rot3(des_data_v_[0].q).rpy();
    float distance_est = (des_data_v_[0].p - pose.translation()).norm();

    std::cout << " ---------- Optimize Time: [ " << opt_cost << " ], " << "ori distance: [ " << distance << " ], est distance: [" << distance_est << " ]" << endl;

    log_ << std::setprecision(19)
         // Des info
         << des_data_v_[0].rcv_stamp.toSec() << " "
         << des_data_v_[0].p.x() << " " << des_data_v_[0].p.y() << " " << des_data_v_[0].p.z() << " "
         << des_data_v_[0].v.x() << " " << des_data_v_[0].v.y() << " " << des_data_v_[0].v.z() << " "
         << des_eular_xyz.x() << " " << des_eular_xyz.y() << " " << des_eular_xyz.z() << " "

         // Positioning GT Info
         << odom.p.x() << " " << odom.p.y() << " " << odom.p.z() << " "
         << gt_eular_xyz.x() << " " << gt_eular_xyz.y() << " " << gt_eular_xyz.z() << " "
         << odom.v.x() << " " << odom.v.y() << " " << odom.v.z() << " "

         // Positioning with Noise
         << odom_noise.p.x() << " " << odom_noise.p.y() << " " << odom_noise.p.z() << " "
         << odom_noise.v.x() << " " << odom_noise.v.y() << " " << odom_noise.v.z() << " "

         // Positioning Estimation
         << pose.translation().x() << " " << pose.translation().y() << " " << pose.translation().z() << " "
         << vel.x() << " " << vel.y() << " " << vel.z() << " "
         << eular_xyz.x() << " " << eular_xyz.y() << " " << eular_xyz.z() << " "

         // Time cost
         << opt_cost << " "

         // IMU Raw Data
         << imu_raw.w.x() << " " << imu_raw.w.y() << " " << imu_raw.w.z() << " "
         << imu_raw.a.x() << " " << imu_raw.a.y() << " " << imu_raw.a.z() << " "

         // Bias
         << imu_bias.accelerometer().x() << " " << imu_bias.accelerometer().y() << " " << imu_bias.accelerometer().z() << " "
         << imu_bias.gyroscope().x() << " " << imu_bias.gyroscope().y() << " " << imu_bias.gyroscope().z() << " "

         // Control
         << thrust2 << " "
         << bodyrates2.x() << " " << bodyrates2.y() << " " << bodyrates2.z() << " " // MPC

         << std::endl;
  }

  if (des_data_v_.size() >= opt_traj_lens_)
  {
    des_data_v_.erase(des_data_v_.begin());
  }

  if (imu_data_v_.size() >= window_lens_)
  {
    // std::cout << " - Start erase imu_data_v_ - " << std::endl;
    imu_data_v_.erase(imu_data_v_.begin());
  }

  if (odom_data_v_.size() >= window_lens_)
  {
    // std::cout << " - Start erase odom_data_v_ - " << std::endl;
    odom_data_v_.erase(odom_data_v_.begin());
  }

  if (odom_data_noise_.size() >= window_lens_)
  {
    // std::cout << " - Start erase odom_data_noise_ - " << std::endl;
    odom_data_noise_.erase(odom_data_noise_.begin());
  }

  return debug_msg_;
}
double Controller::rpy_from_quaternion(const Eigen::Quaterniond &q)
{
  // 更高效的计算公式（避免平方根运算）
  const double w = q.w(), x = q.x(), y = q.y(), z = q.z();

  // 计算偏航角 [-π, π]
  double yaw = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
  return normalize_yaw(yaw);
  // 转换为 [0, 2π] 范围
  // return (yaw >= 0) ? yaw : (yaw + 2.0 * M_PI);
}

double Controller::normalize_yaw(double yaw)
{
  yaw = fmod(yaw, 2 * M_PI);
  if (yaw > M_PI)
    yaw -= 2 * M_PI;
  else if (yaw < -M_PI)
    yaw += 2 * M_PI;
  return yaw;
}
/*
 * Single-point (SP) JPCM
 */
quadrotor_msgs::Px4ctrlDebug Controller::calculateControl(const Desired_State_t &des, const Odom_Data_t &GT, const Odom_Data_t &odom, const Imu_Data_t &imu,
                                                          Controller_Output_t &thr_bodyrate_u, CTRL_MODE mode_switch)
{
  clock_t all_start, all_end;
  all_start = clock();
  Odom_Data_t odom_noise = odom;
  bool timeout = false;
  double all_cost = 0.0f;
  double opt_cost = 0.0f;
  double thrust = 0.0f;
  double thrust2 = 0.0f;
  gtsam::Vector3 bodyrates(0, 0, 0);
  gtsam::Vector3 bodyrates2(0, 0, 0);

  des_data_v_.push_back(des);

  if (timeout || mode_switch == DFBC || des_data_v_.size() < opt_traj_lens_)
  {
    Controller::calculateControl(des_data_v_[0], odom_noise, imu, thr_bodyrate_u);
    thrust = thr_bodyrate_u.thrust;
    bodyrates = thr_bodyrate_u.bodyrates;
    // thr_bodyrate_u.thrust = 0;
    // thr_bodyrate_u.bodyrates = bodyrates;
  }
  else if (mode_switch == MPC && des_data_v_.size() == opt_traj_lens_)
  {
    clock_t start, end;
    gtsam::LevenbergMarquardtParams parameters;
    parameters.absoluteErrorTol = 100;
    parameters.relativeErrorTol = 1e-2;
    parameters.maxIterations = 10;
    parameters.verbosity = gtsam::NonlinearOptimizerParams::SILENT;
    parameters.verbosityLM = gtsam::LevenbergMarquardtParams::SILENT; // SILENT
    graph_.empty();

    // 2. 计算初始误差（优化前的总残差）
    double initialError = graph_.error(initial_value_); // 关键接口：计算初始残差

    FGbuilder->buildFactorGraph(graph_, initial_value_, des_data_v_, odom_noise, dt_);
    LevenbergMarquardtOptimizer optimizer(graph_, initial_value_, parameters);

    start = clock();
    Values result = optimizer.optimize();
    // initial_value_ = result; // 更新初始值为优化后的结果
    end = clock();
    opt_cost = (double)(end - start) / CLOCKS_PER_SEC;
    float distance = (des_data_v_[0].p - odom_noise.p).norm();

    gtsam::Vector4 input;

    input = result.at<gtsam::Vector4>(U(0));
    Eigen::Vector3d des_acc(0, 0, input[0]);

    thrust2 = Controller::computeDesiredCollectiveThrustSignal(des_acc);
    bodyrates2 = Eigen::Vector3d(input[1], input[2], input[3]);

    thr_bodyrate_u.thrust = thrust2;
    thr_bodyrate_u.bodyrates = bodyrates2;

    all_end = clock();

    int iterations = optimizer.iterations(); // 实际迭代步数
    double finalError = optimizer.error();   // 最终残差（优化后的总误差）
    int converged = 0;
    if (finalError <= parameters.absoluteErrorTol ||
        (iterations > 0 && std::abs(finalError - initialError) / initialError <= parameters.relativeErrorTol))
    {
      converged = 1; // 达到绝对/相对误差阈值
    }
    else if (iterations >= parameters.maxIterations)
    {
      converged = 0; // 达到最大迭代次数但未达标
    }
    all_cost = (double)(all_end - all_start) / CLOCKS_PER_SEC;
    log_ << std::fixed << std::setprecision(4)
         << odom_noise.rcv_stamp.toSec() << " "
         << iterations << " " << initialError << " " << finalError << " " << converged << " "
         << opt_cost << " " << all_cost << " "
         << input[0] << " "
         << std::endl;
  }

  if (des_data_v_.size() >= opt_traj_lens_)
  {
    des_data_v_.erase(des_data_v_.begin());
  }

  return debug_msg_;
}

double Controller::fromQuaternion2yaw(Eigen::Quaterniond q)
{
  double yaw = atan2(2 * (q.x() * q.y() + q.w() * q.z()), q.w() * q.w() + q.x() * q.x() - q.y() * q.y() - q.z() * q.z());
  return yaw;
}

Controller::Controller(Parameter_t &param) : param_(param)
{
  resetThrustMapping();
  time_t now = time(NULL);
  tm *t = localtime(&now);
  FGbuilder = std::make_shared<buildJPCMFG>(param);

  graph_.empty();
  dt_ = 0.01f;
  opt_traj_lens_ = param_.factor_graph.OPT_LENS_TRAJ;
  window_lens_ = param_.factor_graph.WINDOW_SIZE;

  if (!param_.output_log_path.empty())
  {
    std::string out_log_file = param_.output_log_path + "controller_log.txt";
    std::cout << " -- log file:" << out_log_file << std::endl;
    log_.open(out_log_file, std::ios::out | std::ios::trunc);
  }

  position_noise_x = Dist_Dou(param_.factor_graph.POS_MEAS_MEAN, param_.factor_graph.POS_MEAS_COV);
  rotation_noise_x = Dist_Dou(0, param_.factor_graph.ROT_MEAS_COV);
  velocity_noise_x = Dist_Dou(0, param_.factor_graph.VEL_MEAS_COV);

  position_noise_y = Dist_Dou(param_.factor_graph.POS_MEAS_MEAN, param_.factor_graph.POS_MEAS_COV);
  rotation_noise_y = Dist_Dou(0, param_.factor_graph.ROT_MEAS_COV);
  velocity_noise_y = Dist_Dou(0, param_.factor_graph.VEL_MEAS_COV);

  position_noise_z = Dist_Dou(param_.factor_graph.POS_MEAS_MEAN, param_.factor_graph.POS_MEAS_COV);
  rotation_noise_z = Dist_Dou(0, param_.factor_graph.ROT_MEAS_COV);
  velocity_noise_z = Dist_Dou(0, param_.factor_graph.VEL_MEAS_COV);

  odom_data_noise_.reserve(window_lens_);
}

/*
 * compute thr_bodyrate_u.thrust and thr_bodyrate_u.q, controller gains and other parameters are in param_
 * Differential-Flatness Based Controller (DFBC) Subject to Aerodynamics Drag Force
 */
quadrotor_msgs::Px4ctrlDebug Controller::calculateControl(const Desired_State_t &des, const Odom_Data_t &odom, const Imu_Data_t &imu,
                                                          Controller_Output_t &thr_bodyrate_u)
{
  /* WRITE YOUR CODE HERE */
  // compute disired acceleration
  Eigen::Vector3d subtract(0, 0, 0);
  if (des.p[2] >= 2.3f)
  {
    subtract = des.p - Eigen::Vector3d(0, 0, 2.3f);
    ROS_WARN("Des.p >= 2.3f");
  }
  gtsam::Rot3 Rc(odom.q);
  Eigen::Vector3d des_acc(0.0, 0.0, 0.0); // des_acc corresponding to collective thrust in the world coordinate system
  Eigen::Vector3d Kp, Kv, KR, KDrag;
  Kp << param_.gain.Kp0, param_.gain.Kp1, param_.gain.Kp2;
  Kv << param_.gain.Kv0, param_.gain.Kv1, param_.gain.Kv2;
  KR << param_.gain.KAngR, param_.gain.KAngP, param_.gain.KAngY;
  KDrag << param_.rt_drag.x, param_.rt_drag.y, param_.rt_drag.z;
  float mass = param_.mass;
  des_acc = des.a + Kv.asDiagonal() * limit_err(des.v - odom.v, param_.gain.VErrMax) + Kp.asDiagonal() * limit_err(des.p - subtract - odom.p, param_.gain.PErrMax);
  des_acc += Eigen::Vector3d(0, 0, param_.gra); // * odom.q * e3
  des_acc += Rc.matrix() * KDrag.asDiagonal() * Rc.inverse().matrix() * odom.v / mass;

  thr_bodyrate_u.thrust = computeDesiredCollectiveThrustSignal(des_acc, odom.v);
  // thr_bodyrate_u.thrust = computeDesiredCollectiveThrustSignal(des_acc);

  Eigen::Vector3d force = des_acc * param_.mass;

  // Limit control angle to 80 degree
  double theta = param_.max_angle;
  double c = cos(theta);
  Eigen::Vector3d f;
  f.noalias() = force - param_.mass * param_.gra * Eigen::Vector3d(0, 0, 1);
  if (Eigen::Vector3d(0, 0, 1).dot(force / force.norm()) < c)
  {
    double nf = f.norm();
    double A = c * c * nf * nf - f(2) * f(2);
    double B = 2 * (c * c - 1) * f(2) * param_.mass * param_.gra;
    double C = (c * c - 1) * param_.mass * param_.mass * param_.gra * param_.gra;
    double s = (-B + sqrt(B * B - 4 * A * C)) / (2 * A);
    force.noalias() = s * f + param_.mass * param_.gra * Eigen::Vector3d(0, 0, 1);
  }

  Eigen::Vector3d b1c, b2c, b3c;
  Eigen::Vector3d b1d(cos(des.yaw), sin(des.yaw), 0);

  if (force.norm() > 1e-6)
    b3c.noalias() = force.normalized();
  else
    b3c.noalias() = Eigen::Vector3d(0, 0, 1);

  b2c.noalias() = b3c.cross(b1d).normalized();
  b1c.noalias() = b2c.cross(b3c).normalized();

  Eigen::Matrix3d R;
  R << b1c, b2c, b3c;

  thr_bodyrate_u.q = Eigen::Quaterniond(R);
  gtsam::Rot3 Rd(thr_bodyrate_u.q);
  thr_bodyrate_u.bodyrates = KR.asDiagonal() * gtsam::Rot3::Logmap(Rc.inverse() * Rd) + des.w;

  debug_msg_.des_v_x = des.v(0);
  debug_msg_.des_v_y = des.v(1);
  debug_msg_.des_v_z = des.v(2);

  debug_msg_.des_a_x = des_acc(0);
  debug_msg_.des_a_y = des_acc(1);
  debug_msg_.des_a_z = des_acc(2);

  debug_msg_.des_q_x = thr_bodyrate_u.q.x();
  debug_msg_.des_q_y = thr_bodyrate_u.q.y();
  debug_msg_.des_q_z = thr_bodyrate_u.q.z();
  debug_msg_.des_q_w = thr_bodyrate_u.q.w();

  debug_msg_.des_thr = thr_bodyrate_u.thrust;

  log_ << std::fixed << std::setprecision(4)
       << odom.rcv_stamp.toSec() << " "
       << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " "
       << 0.0 << " " << 0.0 << " "
       << thr_bodyrate_u.thrust << " "
       << std::endl;
  // Used for thrust-accel mapping estimation
  timed_thrust_.push(std::pair<ros::Time, double>(ros::Time::now(), thr_bodyrate_u.thrust));
  while (timed_thrust_.size() > 100)
  {
    timed_thrust_.pop();
  }
  return debug_msg_;
}
// ! --------------------- 计算期望姿态 ---------------------------------
Eigen::Quaterniond Controller::computeDesiredAttitude(
    const Eigen::Vector3d &desired_acceleration, const double reference_heading,
    const Eigen::Quaterniond &attitude_estimate) const
{
  const Eigen::Quaterniond q_heading = Eigen::Quaterniond(
      Eigen::AngleAxisd(reference_heading, Eigen::Vector3d::UnitZ()));

  // Compute desired orientation
  const Eigen::Vector3d x_C = q_heading * Eigen::Vector3d::UnitX();
  const Eigen::Vector3d y_C = q_heading * Eigen::Vector3d::UnitY();

  Eigen::Vector3d z_B;
  if (almostZero(desired_acceleration.norm()))
  {
    z_B = attitude_estimate * Eigen::Vector3d::UnitZ();
  }
  else
  {
    z_B = desired_acceleration.normalized();
  }

  const Eigen::Vector3d x_B_prototype = y_C.cross(z_B);
  const Eigen::Vector3d x_B =
      computeRobustBodyXAxis(x_B_prototype, x_C, y_C, attitude_estimate);

  const Eigen::Vector3d y_B = (z_B.cross(x_B)).normalized();

  // From the computed desired body axes we can now compose a desired attitude
  const Eigen::Matrix3d R_W_B((Eigen::Matrix3d() << x_B, y_B, z_B).finished());

  const Eigen::Quaterniond desired_attitude(R_W_B);

  return desired_attitude;
}
// ! --------------------- 计算参考输入作为前馈项 -------------------------
Eigen::Vector3d Controller::computeNominalReferenceInputs(
    const Eigen::Quaterniond &attitude_estimate, const Desired_State_t &des) const
{
  double heading = des.yaw;
  double heading_rate = des.yaw_rate;
  Eigen::Vector3d jerk(des.j.x(), des.j.y(), des.j.z());
  Eigen::Vector3d acc(des.a.x(), des.a.y(), des.a.z());

  const Eigen::Quaterniond q_heading = Eigen::Quaterniond(
      Eigen::AngleAxisd(heading, Eigen::Vector3d::UnitZ()));

  const Eigen::Vector3d x_C = q_heading * Eigen::Vector3d::UnitX();
  const Eigen::Vector3d y_C = q_heading * Eigen::Vector3d::UnitY();

  const Eigen::Vector3d des_acc = acc + Eigen::Vector3d(0, 0, param_.gra);

  // Reference attitude
  const Eigen::Quaterniond q_W_B = computeDesiredAttitude(
      des_acc, heading, attitude_estimate);

  const Eigen::Vector3d x_B = q_W_B * Eigen::Vector3d::UnitX();
  const Eigen::Vector3d y_B = q_W_B * Eigen::Vector3d::UnitY();
  const Eigen::Vector3d z_B = q_W_B * Eigen::Vector3d::UnitZ();

  Eigen::Quaterniond orientation = q_W_B;

  Eigen::Vector3d bodyrates;
  Eigen::Vector3d angular_accelerations;
  double collective_thrust;

  // Reference thrust
  collective_thrust = des_acc.norm();

  // Reference body rates
  if (almostZeroThrust(collective_thrust))
  {
    bodyrates.x() = 0.0;
    bodyrates.y() = 0.0;
  }
  else
  {
    bodyrates.x() = -1.0 /
                    collective_thrust *
                    y_B.dot(jerk);
    bodyrates.y() = 1.0 /
                    collective_thrust *
                    x_B.dot(jerk);
  }

  if (almostZero((y_C.cross(z_B)).norm()))
  {
    bodyrates.z() = 0.0;
  }
  else
  {
    bodyrates.z() =
        1.0 / (y_C.cross(z_B)).norm() *
        (heading_rate * x_C.dot(x_B) +
         bodyrates.y() * y_C.dot(z_B));
  }
  return bodyrates;
}

// ! --------------------- 安全措施辅助函数   ----------------------------
bool Controller::almostZeroThrust(const double thrust_value) const
{
  return fabs(thrust_value) < 0.01;
}

Eigen::Vector3d Controller::computeRobustBodyXAxis(
    const Eigen::Vector3d &x_B_prototype, const Eigen::Vector3d &x_C,
    const Eigen::Vector3d &y_C,
    const Eigen::Quaterniond &attitude_estimate) const
{

  Eigen::Vector3d x_B = x_B_prototype;

  if (almostZero(x_B.norm()))
  {
    const Eigen::Vector3d x_B_estimated =
        attitude_estimate * Eigen::Vector3d::UnitX();
    const Eigen::Vector3d x_B_projected =
        x_B_estimated - (x_B_estimated.dot(y_C)) * y_C;
    if (almostZero(x_B_projected.norm()))
    {
      x_B = x_C;
    }
    else
    {
      x_B = x_B_projected.normalized();
    }
  }
  else
  {
    x_B.normalize();
  }
  return x_B;
}

bool Controller::almostZero(const double value) const
{
  return fabs(value) < 0.001;
}
/*
  compute throttle percentage
*/
double Controller::computeDesiredCollectiveThrustSignal(const Eigen::Vector3d &des_acc, const Eigen::Vector3d &v)
{
  double throttle_percentage(0.0);

  /* compute throttle, thr2acc has been estimated before */
  throttle_percentage = (des_acc.norm() - param_.rt_drag.k_thrust_horz * (pow(v.x(), 2.0) + pow(v.y(), 2.0)) / param_.mass) / thr2acc_;
  throttle_percentage = limit_value(param_.thr_map.thrust_upper_bound, throttle_percentage, param_.thr_map.thrust_lower_bound);
  return throttle_percentage;
}

/*
  compute throttle percentage
*/
double Controller::computeDesiredCollectiveThrustSignal(const Eigen::Vector3d &des_acc)
{
  double throttle_percentage(0.0);

  /* compute throttle, thr2acc has been estimated before */
  // throttle_percentage = des_acc.norm() / thr2acc_;
  throttle_percentage = (des_acc.norm() * param_.thr_map.hover_percentage) / (param_.mass * param_.gra);
  throttle_percentage = limit_value(param_.thr_map.thrust_upper_bound, throttle_percentage, param_.thr_map.thrust_lower_bound);
  return throttle_percentage;
}
// 测量值：est_a(2) = thr2acc × thr + ε
// est_a(2) : Z轴加速度测量值（m / s²）
// thr : 油门指令值（归一化0 ~1）
// thr2acc : 待估计的油门
// - 加速度转换系数
// ε : 测量噪声
bool Controller::estimateThrustModel(const Eigen::Vector3d &est_a, const Parameter_t &param)
{
  ros::Time t_now = ros::Time::now();
  while (timed_thrust_.size() >= 1)
  {
    // Choose thrust data before 35~45ms ago
    std::pair<ros::Time, double> t_t = timed_thrust_.front();
    double time_passed = (t_now - t_t.first).toSec();
    if (time_passed > 0.045) // 45ms
    {
      // printf("continue, time_passed=%f\n", time_passed);
      timed_thrust_.pop();
      continue;
    }
    if (time_passed < 0.035) // 35ms
    {
      // printf("skip, time_passed=%f\n", time_passed);
      return false;
    }

    /***********************************************************/
    /* Recursive least squares algorithm with vanishing memory */
    /***********************************************************/
    double thr = t_t.second;
    timed_thrust_.pop();

    /***********************************/
    /* Model: est_a(2) = thr2acc_ * thr */
    /***********************************/
    double gamma = 1 / (rho2_ + thr * P_ * thr);
    double K = gamma * P_ * thr;
    thr2acc_ = thr2acc_ + K * (est_a(2) - thr * thr2acc_);
    P_ = (1 - K * thr) * P_ / rho2_;
    // printf("Thrust debug [ thr2acc: %6.3f, gamma: %6.3f, K: %6.3f, P: %6.3f, thrust: %6.3f, est_a(2): %6.3f ]\n", thr2acc_, gamma, K, P_, thr, est_a(2));
    // debug_msg_.thr2acc = thr2acc_;
    return true;
  }
  return false;
}

void Controller::resetThrustMapping(void)
{
  thr2acc_ = param_.gra / param_.thr_map.hover_percentage;
  P_ = 1e6;
}

double Controller::limit_value(double upper_bound, double input, double lower_bound)
{
  if (upper_bound <= lower_bound)
  {
    log_ << "Warning: upper_bound <= lower_bound\n";
  }
  if (input > upper_bound)
  {
    input = upper_bound;
  }
  if (input < lower_bound)
  {
    input = lower_bound;
  }
  return input;
}

Controller::~Controller()
{
  log_.close();
}

Eigen::Vector3d Controller::limit_err(const Eigen::Vector3d err, const double p_err_max)
{
  Eigen::Vector3d r_err(0, 0, 0);
  for (uint i = 0; i < 3; i++)
  {
    r_err[i] = limit_value(std::abs(p_err_max), err[i], -std::abs(p_err_max));
  }
  return r_err;
}

Odom_Data_t Controller::add_Guassian_noise(const Odom_Data_t &odom)
{
  gtsam::Vector3 pos_noise = gtsam::Vector3(position_noise_x(__randomGen), position_noise_y(__randomGen), position_noise_z(__randomGen));
  gtsam::Vector3 vel_noise = gtsam::Vector3(velocity_noise_x(__randomGen), velocity_noise_y(__randomGen), velocity_noise_z(__randomGen));
  gtsam::Vector3 rot_noise = gtsam::Vector3(rotation_noise_x(__randomGen), rotation_noise_y(__randomGen), rotation_noise_z(__randomGen));
  gtsam::Vector3 rot_add = gtsam::Rot3::Logmap(gtsam::Rot3(odom.q)) + rot_noise;
  gtsam::Rot3 rot3_add = gtsam::Rot3::Expmap(rot_add);

  Odom_Data_t odom_noise;
  odom_noise.rcv_stamp = odom.rcv_stamp;
  odom_noise.p = odom.p + pos_noise;
  odom_noise.v = odom.v + vel_noise;
  odom_noise.q = Eigen::Quaterniond(rot3_add.toQuaternion().w(), rot3_add.toQuaternion().x(), rot3_add.toQuaternion().y(), rot3_add.toQuaternion().z());

  return odom_noise;
}

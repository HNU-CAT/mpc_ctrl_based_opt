# mpc_ctrl_based_opt 项目介绍

## 项目概述
`mpc_ctrl_based_opt` 是一个基于模型预测控制（MPC）的优化控制项目，主要面向无人机等移动设备的控制与导航场景。项目基于ROS（机器人操作系统）开发，集成了控制算法实现、可视化工具、位姿处理及消息定义等功能模块，提供了从算法到可视化的完整解决方案。

## 核心功能模块

### 控制核心模块
- **mpc_ctrl**：项目核心功能包，实现模型预测控制相关算法
  - 包含 `JPCM_node` 主节点，负责控制逻辑执行
  - 集成 GTSAM 库进行状态估计与优化（包含因子定义、边缘化等模块）
  - 支持轨迹测试脚本（{insert\_element\_1\_YGZpeGVkX3BvaW50X2FuZF9jaXJjbGVfdGVzdC5weWA=}）

## 系统依赖
- 操作系统：Linux（兼容ROS支持的发行版）
- 核心框架：ROS（支持Indigo等版本）
- 构建系统：catkin
- 主要库依赖：
  - Eigen3：矩阵运算
  - GTSAM 4.0.3：状态估计与优化
  - YAML-CPP：配置文件处理

## 编译与使用
1. 将仓库克隆到ROS工作空间的`src`目录
   ```bash
   cd ~/catkin_ws/src
   git clone <仓库地址>
   
## 编译工作空间
1. 将仓库克隆到ROS工作空间的`src`目录
   ```bash
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash

## 运行核心功能(Sim)
1. 启动您的px4 Sim
   ```bash
   roslaunch px4 indoor1.launch
2. 启动控制器
   ```bash
   roslaunch mpc_ctrl run_ctrl.launch
3. 启动轨迹生成器
   ```bash
   roslaunch mpc_ctrl pub_cmd.launch
3. mpc控制效果可视化脚本
   ```bash
   python3 ~/catkin_ws/src/your_pkg/mpc_ctrl/scripts/controller_vis.py  # 轨迹跟踪结果分析可视化
   python3 ~/catkin_ws/src/your_pkg/mpc_ctrl/scripts/opt_vis.py # 优化结果分析可视化

## 关键参数介绍
1. controller_param.yaml主要负责轨迹生成
   ```bash
   real_world_flag: False # Set to true if you are running on a real-world drone, false for simulation.
   odom_topic: /mavros/local_position/odom # your odom topic
   imu_topic: /mavros/imu/data # your imu topic, hz > 100Hz
   target_topic: /planning/pos_cmd # target traj topic
   traj_mode: 0 # 0: fixed position control; 1: circle control
   attitude_target_topic: /mavros/setpoint_raw/attitude
   position_target_topic: /mavros/setpoint_raw/local
   save_path: ~/experiment/mpc_ctrl_ws/src/mpc_ctrl/scripts/mpc_tracking_data.csv # please sure path in your_pkg/scripts
2. ctrl_param_fpv.yaml主要负责控制器参数
   ```bash
   mass: 1.8065 # kg
   gra: 9.81
   ctrl_freq_max: 100.0
   use_bodyrate_ctrl: true
   max_angle: 80 # Attitude angle limit in degree. A negative value means no limit.
   odom_freq: 100
   ctrl_mode: 2 # DFBC 1 MPC 2 JPCM 3

   odom_topic: /mavros/local_position/odom

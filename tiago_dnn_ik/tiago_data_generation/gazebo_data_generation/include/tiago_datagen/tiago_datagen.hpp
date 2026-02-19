#ifndef __tiago_data_generation_HPP__
#define __tiago_data_generation_HPP__

#include <array>
#include <mutex>
#include <utility> // std::pair
#include <vector>
#include <random>

#include <iostream>
#include <fstream>
#include <string>

#include <controller_interface/controller.h>
#include <hardware_interface/joint_command_interface.h>

#include <kdl/chain.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/jntarray.hpp>

namespace tiago_controllers
{

class DatagenController : public controller_interface::Controller<hardware_interface::PositionJointInterface>
{
public:
  ~DatagenController();

  bool init(hardware_interface::PositionJointInterface* hw, ros::NodeHandle &n);
  void update(const ros::Time& time, const ros::Duration& period);
  void starting(const ros::Time& time);
  void stopping(const ros::Time& time);

private:
  bool checkReturnCode(int ret);

  std::vector<hardware_interface::JointHandle> armJoints;
  std::vector<std::pair<double, double>> armJointLimits;

  KDL::Chain chain;
  KDL::ChainFkSolverPos * fkSolverPos {nullptr};

  KDL::JntArray q;
  KDL::JntArray q_temp;

  int num_pos;
  int num = 0;
  std::string orient;

  std::vector<std::uniform_real_distribution<float>> distrs;
  std::string filename;
  std::ofstream file;

  ros::Time begin;

  bool dual;
};

} //namespace

#endif // __tiago_data_generation_HPP__

#include "tiago_datagen/tiago_datagen.hpp"

#include <algorithm> // std::copy
#include <string>
#include <pluginlib/class_list_macros.h>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/frames.hpp>
#include <urdf/model.h>

using namespace tiago_controllers;

constexpr auto UPDATE_LOG_THROTTLE = 1.0; // [s]

DatagenController::~DatagenController()
{

}

bool DatagenController::init(hardware_interface::PositionJointInterface* hw, ros::NodeHandle &n)
{
    KDL::Tree tree;
    std::string robot_desc_string;

    if (!n.getParam("/robot_description", robot_desc_string))
    {
        ROS_ERROR("Could not find robot_description");
        return false;
    }

    if (!kdl_parser::treeFromString(robot_desc_string, tree))
    {
        ROS_ERROR("Failed to construct kdl tree");
        return false;
    }

    // TIAGo or TIAGo++
    if (!n.getParam("dual", dual))
    {
        ROS_ERROR("Could not find dual parameter");
        return false;
    }

    std::string start_link, end_link;

    if (!n.getParam("start_link", start_link))
    {
        ROS_ERROR("Could not find start_link parameter");
        return false;
    }

    if (dual)
    {
        if (!n.getParam("dual_end_link", end_link))
        {
            ROS_ERROR("Could not find dual_end_link parameter");
            return false;
        }
    }
    else 
    {
        if (!n.getParam("end_link", end_link))
        {
            ROS_ERROR("Could not find end_link parameter");
            return false;
        }
    }

    if (!tree.getChain(start_link, end_link, chain))
    {
        ROS_ERROR("Failed to get chain from kdl tree");
        return false;
    }

    ROS_INFO("Got chain with %d joints and %d segments", chain.getNrOfJoints(), chain.getNrOfSegments());

    q.resize(chain.getNrOfJoints());
    q_temp.resize(chain.getNrOfJoints());

    fkSolverPos = new KDL::ChainFkSolverPos_recursive(chain);

    urdf::Model model;

    if (!model.initString(robot_desc_string))
    {
        ROS_ERROR("Failed to parse urdf file");
        return false;
    }

    std::vector<std::string> arm_joint_names;

    if (dual)
    {
        if (!n.getParam("dual_joint_names", arm_joint_names))
        {
            ROS_ERROR("Could not retrieve dual arm joint names");
            return false;
        }
    }
    else
    {
        if (!n.getParam("arm_joint_names", arm_joint_names))
        {
            ROS_ERROR("Could not retrieve arm joint names");
            return false;
        }
    }

    for (const auto & joint_name : arm_joint_names)
    {
        armJoints.push_back(hw->getHandle(joint_name));

        armJointLimits.emplace_back(model.getJoint(joint_name)->limits->lower,
                                    model.getJoint(joint_name)->limits->upper);

        std::uniform_real_distribution<float> distr(model.getJoint(joint_name)->limits->lower, 
                                                    model.getJoint(joint_name)->limits->upper);
        distrs.push_back(distr);
    }

    if (!n.getParam("num_pos", num_pos))
    {
        ROS_ERROR("Could not find num_pos parameter");
        return false;
    }

    if (!n.getParam("orient", orient))
    {
        ROS_ERROR("Could not find orientation parameter");
        return false;
    }

    if (!n.getParam("filename", filename))
    {
        ROS_ERROR("Could not find filename parameter");
        return false;
    }

    file.open(filename);
    if (!file.is_open()) 
    {
        std::cerr << "Failed to open file!" << std::endl;
        return 1;
    }

    if (orient == "quaternion") 
    {
        file << "arm_1,arm_2,arm_3,arm_4,arm_5,arm_6,arm_7,ee_x,ee_y,ee_z,ee_quat_x,ee_quat_y,ee_quat_z,ee_quat_w\n";    
    }
    else if (orient == "matrix") 
    {
        file << "arm_1,arm_2,arm_3,arm_4,arm_5,arm_6,arm_7,ee_x,ee_y,ee_z,ee_rot_00,ee_rot_01,ee_rot_02,ee_rot_10,ee_rot_11,ee_rot_12,ee_rot_20,ee_rot_21,ee_rot_22\n";
    }
    else 
    {
        file << "arm_1,arm_2,arm_3,arm_4,arm_5,arm_6,arm_7,ee_x,ee_y,ee_z\n";
    }

    begin = ros::Time::now();

    return true;
}

void DatagenController::update(const ros::Time& time, const ros::Duration& period)
{
    KDL::Frame H_root_ee;
    std::string out = "";

    std::random_device rd;
    std::mt19937 e2(rd());

    std::vector<float> data;

    if (num < num_pos)
    {
        // get joint positions
        for (int i = 0; i < armJoints.size(); i++)
        {
            q(i) = armJoints[i].getPosition();
            data.push_back(q(i));
        }

        ROS_INFO("%d", num);

        // forward kinematics to get position from end effector
        if (!checkReturnCode(fkSolverPos->JntToCart(q, H_root_ee)))
        {
            return;
        }
        data.push_back(H_root_ee.p[0]);
        data.push_back(H_root_ee.p[1]);
        data.push_back(H_root_ee.p[2]);

        // get orientation
        if (orient == "quaternion") {
            double x, y, z, w;
            H_root_ee.M.GetQuaternion(x, y, z, w);
            data.push_back(x);
            data.push_back(y);
            data.push_back(z);
            data.push_back(w);
        }

        else if (orient == "matrix") {
            data.push_back(H_root_ee.M(0, 0));
            data.push_back(H_root_ee.M(0, 1));
            data.push_back(H_root_ee.M(0, 2));
            data.push_back(H_root_ee.M(1, 0));
            data.push_back(H_root_ee.M(1, 1));
            data.push_back(H_root_ee.M(1, 2));
            data.push_back(H_root_ee.M(2, 0));
            data.push_back(H_root_ee.M(2, 1));
            data.push_back(H_root_ee.M(2, 2));
        }

        // workspace size filters
        // TODO: root is torso lift link, so the floor is not at 0 but lower
        if (H_root_ee.p[2] < 0.0)
        {
            ROS_WARN("Too close to the floor!");
        }
        if (H_root_ee.p[1] < -0.3 || H_root_ee.p[1] > 0.06)
        {
            ROS_WARN("Too far sideways!");
        }
        if (H_root_ee.p[0] < 0.4 || H_root_ee.p[0] > 0.7)
        {
            ROS_WARN("Too far out!");
        }
        else {
            // write data to csv
            for (int i = 0; i < data.size(); i++)
            {
                file << data[i];
                if (i != data.size() - 1) 
                    file << ",";
            }
            file << "\n";

            num = num + 1;
        }

        // generate random pose
        for (int i = 0; i < armJoints.size(); i++)
        {
            if (std::abs(q(i) - q_temp(i)) < 0.05)
            {
                std::uniform_real_distribution<float> dist = distrs[i];
                q_temp(i) = dist(e2);
                armJoints[i].setCommand(q_temp(i));
            }
        }
    }

    else
    {
        if (file.is_open()) 
        {
            file.close();
            std::cout << "CSV file " << filename << " created successfully." << std::endl;
            ros::Duration duration = ros::Time::now() - begin;
            ROS_INFO("Generation time: %lf secs", duration.toSec());
        }
    }
}

bool DatagenController::checkReturnCode(int ret)
{
    switch (ret)
    {
        case KDL::SolverI::E_SIZE_MISMATCH:
            ROS_WARN_THROTTLE(UPDATE_LOG_THROTTLE, "Size mismatch: child fkSolverPos failed");
            return false;
        case KDL::SolverI::E_OUT_OF_RANGE:
            ROS_WARN_THROTTLE(UPDATE_LOG_THROTTLE, "Segment number out of range: child fkSolverPos failed");
            return false;
        case KDL::SolverI::E_MAX_ITERATIONS_EXCEEDED:
            ROS_ERROR_THROTTLE(UPDATE_LOG_THROTTLE, "Convergence issue: maxiter exceeded");
            return false;
        case KDL::SolverI::E_NOERROR:
            return true;
        default:
            ROS_WARN_THROTTLE(UPDATE_LOG_THROTTLE, "Convergence issue: unknown error");
            return false;
    }
}

void DatagenController::starting(const ros::Time& time)
{
    std::string out = "Initial arm pose:";

    for (int i = 0; i < armJoints.size(); i++)
    {
        q(i) = armJoints[i].getPosition();
        out += " " + std::to_string(q(i));
    }

    ROS_INFO("%s", out.c_str());
}

void DatagenController::stopping(const ros::Time& time)
{ }

PLUGINLIB_EXPORT_CLASS(tiago_controllers::DatagenController, controller_interface::ControllerBase);

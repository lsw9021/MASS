#ifndef __DART_HELPER_H__
#define __DART_HELPER_H__
#include "dart/dart.hpp"
namespace Eigen {

using Vector1d = Matrix<double, 1, 1>;
using Matrix1d = Matrix<double, 1, 1>;
}
std::vector<double> split_to_double(const std::string& input, int num);
Eigen::Vector1d string_to_vector1d(const std::string& input);
Eigen::Vector3d string_to_vector3d(const std::string& input);
Eigen::Vector4d string_to_vector4d(const std::string& input);
Eigen::VectorXd string_to_vectorXd(const std::string& input, int n);
Eigen::Matrix3d string_to_matrix3d(const std::string& input);
namespace MASS
{
dart::dynamics::ShapePtr MakeSphereShape(double radius);
dart::dynamics::ShapePtr MakeBoxShape(const Eigen::Vector3d& size);
dart::dynamics::ShapePtr MakeCapsuleShape(double radius, double height);

dart::dynamics::Inertia MakeInertia(const dart::dynamics::ShapePtr& shape,double mass);

dart::dynamics::FreeJoint::Properties* MakeFreeJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity());
dart::dynamics::PlanarJoint::Properties* MakePlanarJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity());
dart::dynamics::BallJoint::Properties* MakeBallJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Vector3d& lower = Eigen::Vector3d::Constant(-2.0),const Eigen::Vector3d& upper = Eigen::Vector3d::Constant(2.0));
dart::dynamics::RevoluteJoint::Properties* MakeRevoluteJointProperties(const std::string& name,const Eigen::Vector3d& axis,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Vector1d& lower = Eigen::Vector1d::Constant(-2.0),const Eigen::Vector1d& upper = Eigen::Vector1d::Constant(2.0));
dart::dynamics::WeldJoint::Properties* MakeWeldJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity());

dart::dynamics::BodyNode* MakeBodyNode(const dart::dynamics::SkeletonPtr& skeleton,dart::dynamics::BodyNode* parent,dart::dynamics::Joint::Properties* joint_properties,const std::string& joint_type,dart::dynamics::Inertia inertia);
dart::dynamics::SkeletonPtr BuildFromFile(const std::string& path,bool create_obj=false);
};

#endif
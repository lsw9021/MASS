#include "DARTHelper.h"
#include <tinyxml.h>
using namespace dart::dynamics;

ShapePtr
MASS::
MakeSphereShape(double radius)
{
	return std::shared_ptr<SphereShape>(new SphereShape(radius));
}
ShapePtr
MASS::
MakeBoxShape(const Eigen::Vector3d& size)
{
	return std::shared_ptr<BoxShape>(new BoxShape(size));
}
ShapePtr
MASS::
MakeCapsuleShape(double radius, double height)
{
	return std::shared_ptr<CapsuleShape>(new CapsuleShape(radius,height));
}

dart::dynamics::Inertia
MASS::
MakeInertia(const dart::dynamics::ShapePtr& shape,double mass)
{
	dart::dynamics::Inertia inertia;

	inertia.setMass(mass);
	inertia.setMoment(shape->computeInertia(mass));
	return inertia;
}

FreeJoint::Properties*
MASS::
MakeFreeJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint,const Eigen::Isometry3d& child_to_joint)
{
	FreeJoint::Properties* props = new FreeJoint::Properties();

	props->mName = name;
	props->mT_ParentBodyToJoint = parent_to_joint;
	props->mT_ChildBodyToJoint = child_to_joint;
	props->mIsPositionLimitEnforced = false;
	props->mVelocityLowerLimits = Eigen::Vector6d::Constant(-100.0);
	props->mVelocityUpperLimits = Eigen::Vector6d::Constant(100.0);
	props->mDampingCoefficients = Eigen::Vector6d::Constant(0.4);

	return props;
}
PlanarJoint::Properties*
MASS::
MakePlanarJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint,const Eigen::Isometry3d& child_to_joint)
{
	PlanarJoint::Properties* props = new PlanarJoint::Properties();

	props->mName = name;
	props->mT_ParentBodyToJoint = parent_to_joint;
	props->mT_ChildBodyToJoint = child_to_joint;
	props->mIsPositionLimitEnforced = false;
	props->mVelocityLowerLimits = Eigen::Vector3d::Constant(-100.0);
	props->mVelocityUpperLimits = Eigen::Vector3d::Constant(100.0);
	props->mDampingCoefficients = Eigen::Vector3d::Constant(0.4);

	return props;
}
BallJoint::Properties*
MASS::
MakeBallJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint,const Eigen::Isometry3d& child_to_joint,const Eigen::Vector3d& lower,const Eigen::Vector3d& upper)
{
	BallJoint::Properties* props = new BallJoint::Properties();

	props->mName = name;
	props->mT_ParentBodyToJoint = parent_to_joint;
	props->mT_ChildBodyToJoint = child_to_joint;
	props->mIsPositionLimitEnforced = true;
	props->mPositionLowerLimits = lower;
	props->mPositionUpperLimits = upper;
	props->mVelocityLowerLimits = Eigen::Vector3d::Constant(-100.0);
	props->mVelocityUpperLimits = Eigen::Vector3d::Constant(100.0);
	props->mForceLowerLimits = Eigen::Vector3d::Constant(-1000.0); 
	props->mForceUpperLimits = Eigen::Vector3d::Constant(1000.0);
	props->mDampingCoefficients = Eigen::Vector3d::Constant(0.4);

	return props;
}
RevoluteJoint::Properties*
MASS::
MakeRevoluteJointProperties(const std::string& name,const Eigen::Vector3d& axis,const Eigen::Isometry3d& parent_to_joint,const Eigen::Isometry3d& child_to_joint,const Eigen::Vector1d& lower,const Eigen::Vector1d& upper)
{
	RevoluteJoint::Properties* props = new RevoluteJoint::Properties();

	props->mName = name;
	props->mT_ParentBodyToJoint = parent_to_joint;
	props->mT_ChildBodyToJoint = child_to_joint;
	props->mIsPositionLimitEnforced = true;
	props->mPositionLowerLimits = lower;
	props->mPositionUpperLimits = upper;
	props->mAxis = axis;
	props->mVelocityLowerLimits = Eigen::Vector1d::Constant(-100.0);
	props->mVelocityUpperLimits = Eigen::Vector1d::Constant(100.0);
	props->mForceLowerLimits = Eigen::Vector1d::Constant(-1000.0); 
	props->mForceUpperLimits = Eigen::Vector1d::Constant(1000.0);
	props->mDampingCoefficients = Eigen::Vector1d::Constant(0.4);

	return props;
}
WeldJoint::Properties*
MASS::
MakeWeldJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint,const Eigen::Isometry3d& child_to_joint)
{
	WeldJoint::Properties* props = new WeldJoint::Properties();

	props->mName = name;
	props->mT_ParentBodyToJoint = parent_to_joint;
	props->mT_ChildBodyToJoint = child_to_joint;

	return props;
}
BodyNode*
MASS::
MakeBodyNode(const SkeletonPtr& skeleton,BodyNode* parent,Joint::Properties* joint_properties,const std::string& joint_type,dart::dynamics::Inertia inertia)
{
	BodyNode* bn;

	if(joint_type == "Free")
	{
		FreeJoint::Properties* prop = dynamic_cast<FreeJoint::Properties*>(joint_properties);
		bn = skeleton->createJointAndBodyNodePair<FreeJoint>(
			parent,(*prop),BodyNode::AspectProperties(joint_properties->mName)).second;
	}
	else if(joint_type == "Planar")
	{
		PlanarJoint::Properties* prop = dynamic_cast<PlanarJoint::Properties*>(joint_properties);
		bn = skeleton->createJointAndBodyNodePair<PlanarJoint>(
			parent,(*prop),BodyNode::AspectProperties(joint_properties->mName)).second;
	}
	else if(joint_type == "Ball")
	{
		BallJoint::Properties* prop = dynamic_cast<BallJoint::Properties*>(joint_properties);
		bn = skeleton->createJointAndBodyNodePair<BallJoint>(
			parent,(*prop),BodyNode::AspectProperties(joint_properties->mName)).second;
	}
	else if(joint_type == "Revolute")
	{
		RevoluteJoint::Properties* prop = dynamic_cast<RevoluteJoint::Properties*>(joint_properties);
		bn = skeleton->createJointAndBodyNodePair<RevoluteJoint>(
			parent,(*prop),BodyNode::AspectProperties(joint_properties->mName)).second;
	}
	else if(joint_type == "Weld")
	{
		WeldJoint::Properties* prop = dynamic_cast<WeldJoint::Properties*>(joint_properties);
		bn = skeleton->createJointAndBodyNodePair<WeldJoint>(
			parent,(*prop),BodyNode::AspectProperties(joint_properties->mName)).second;
	}

	bn->setInertia(inertia);
	return bn;
}
Eigen::Vector3d Proj(const Eigen::Vector3d& u,const Eigen::Vector3d& v)
{
	Eigen::Vector3d proj;
	proj = u.dot(v)/u.dot(u)*u;
	return proj;	
}
Eigen::Isometry3d Orthonormalize(const Eigen::Isometry3d& T_old)
{
	Eigen::Isometry3d T;
	T.translation() = T_old.translation();
	Eigen::Vector3d v0,v1,v2;
	Eigen::Vector3d u0,u1,u2;
	v0 = T_old.linear().col(0);
	v1 = T_old.linear().col(1);
	v2 = T_old.linear().col(2);

	u0 = v0;
	u1 = v1 - Proj(u0,v1);
	u2 = v2 - Proj(u0,v2) - Proj(u1,v2);

	u0.normalize();
	u1.normalize();
	u2.normalize();

	T.linear().col(0) = u0;
	T.linear().col(1) = u1;
	T.linear().col(2) = u2;
	return T;
}
std::vector<double> split_to_double(const std::string& input, int num)
{
    std::vector<double> result;
    std::string::size_type sz = 0, nsz = 0;
    for(int i = 0; i < num; i++){
        result.push_back(std::stof(input.substr(sz), &nsz));
        sz += nsz;
    }
    return result;
}
Eigen::Vector1d string_to_vector1d(const std::string& input){
	std::vector<double> v = split_to_double(input, 1);
	Eigen::Vector1d res;
	res << v[0];

	return res;
}
Eigen::Vector3d string_to_vector3d(const std::string& input){
	std::vector<double> v = split_to_double(input, 3);
	Eigen::Vector3d res;
	res << v[0], v[1], v[2];

	return res;
}
Eigen::Vector4d string_to_vector4d(const std::string& input)
{
	std::vector<double> v = split_to_double(input, 4);
	Eigen::Vector4d res;
	res << v[0], v[1], v[2],v[3];

	return res;	
}
Eigen::VectorXd string_to_vectorXd(const std::string& input, int n){
	std::vector<double> v = split_to_double(input, n);
	Eigen::VectorXd res(n);
	for(int i = 0; i < n; i++){
		res[i] = v[i];
	}

	return res;
}
Eigen::Matrix3d string_to_matrix3d(const std::string& input){
	std::vector<double> v = split_to_double(input, 9);
	Eigen::Matrix3d res;
	res << v[0], v[1], v[2],
			v[3], v[4], v[5],
			v[6], v[7], v[8];

	return res;
}
dart::dynamics::SkeletonPtr
MASS::
BuildFromFile(const std::string& path,bool create_obj)
{
	TiXmlDocument doc;
	if(!doc.LoadFile(path)){
		std::cout << "Can't open file : " << path << std::endl;
		return nullptr;
	}

	TiXmlElement *skeleton_elem = doc.FirstChildElement("Skeleton");
	std::string skel_name = skeleton_elem->Attribute("name");
	SkeletonPtr skel = Skeleton::create(skel_name);
	std::cout << skel_name;

	for(TiXmlElement* node = skeleton_elem->FirstChildElement("Node");node != nullptr;node = node->NextSiblingElement("Node"))
	{
		std::string name = node->Attribute("name");
		std::string parent_str = node->Attribute("parent");
		BodyNode* parent = nullptr;
		if(parent_str != "None")
			parent = skel->getBodyNode(parent_str);
		
		ShapePtr shape;
		Eigen::Isometry3d T_body = Eigen::Isometry3d::Identity();
		TiXmlElement* body = node->FirstChildElement("Body");
		std::string type = body->Attribute("type");
		std::string obj_file = "None";
		if(body->Attribute("obj"))
			obj_file = body->Attribute("obj");
		double mass = std::stod(body->Attribute("mass"));

		if(type == "Box")
		{
			Eigen::Vector3d size = string_to_vector3d(body->Attribute("size"));
			shape = MASS::MakeBoxShape(size);
		}
		else if(type=="Sphere")
		{
			double radius = std::stod(body->Attribute("radius"));
			shape = MASS::MakeSphereShape(radius);
		}
		else if(type=="Capsule")
		{
			double radius = std::stod(body->Attribute("radius"));
			double height = std::stod(body->Attribute("height"));
			shape = MASS::MakeCapsuleShape(radius,height);
		}
		bool contact = false;
		if(body->Attribute("contact")!=nullptr){
			std::string c = body->Attribute("contact");
			if(c == "On")
				contact = true;
		}
		Eigen::Vector4d color = Eigen::Vector4d::Constant(0.2);
		if(body->Attribute("color")!=nullptr)
			color = string_to_vector4d(body->Attribute("color"));

		dart::dynamics::Inertia inertia = MakeInertia(shape,mass);
		T_body.linear() = string_to_matrix3d(body->FirstChildElement("Transformation")->Attribute("linear"));
		T_body.translation() = string_to_vector3d(body->FirstChildElement("Transformation")->Attribute("translation"));
		T_body = Orthonormalize(T_body);			
		TiXmlElement* joint = node->FirstChildElement("Joint");
		type = joint->Attribute("type");
		Joint::Properties* props;
		Eigen::Isometry3d T_joint = Eigen::Isometry3d::Identity();
		T_joint.linear() = string_to_matrix3d(joint->FirstChildElement("Transformation")->Attribute("linear"));
		T_joint.translation() = string_to_vector3d(joint->FirstChildElement("Transformation")->Attribute("translation"));
		T_joint = Orthonormalize(T_joint);	
		Eigen::Isometry3d parent_to_joint;
		if(parent==nullptr)
			parent_to_joint = T_joint;
		else
			parent_to_joint = parent->getTransform().inverse()*T_joint;
		Eigen::Isometry3d child_to_joint = T_body.inverse()*T_joint;
		if(type == "Free")
		{
			props = MASS::MakeFreeJointProperties(name,parent_to_joint,child_to_joint);
		}
		else if(type == "Planar")
		{
			props = MASS::MakePlanarJointProperties(name,parent_to_joint,child_to_joint);
		}
		else if(type == "Ball")
		{
			Eigen::Vector3d lower = string_to_vector3d(joint->Attribute("lower"));
			Eigen::Vector3d upper = string_to_vector3d(joint->Attribute("upper"));
			props = MASS::MakeBallJointProperties(name,parent_to_joint,child_to_joint,lower,upper);
		}
		else if(type == "Revolute")
		{
			Eigen::Vector1d lower = string_to_vector1d(joint->Attribute("lower"));
			Eigen::Vector1d upper = string_to_vector1d(joint->Attribute("upper"));
			Eigen::Vector3d axis = string_to_vector3d(joint->Attribute("axis"));
			props = MASS::MakeRevoluteJointProperties(name,axis,parent_to_joint,child_to_joint,lower,upper);
		}
		else if(type == "Weld")
		{
			props = MASS::MakeWeldJointProperties(name,parent_to_joint,child_to_joint);
		}

		auto bn = MakeBodyNode(skel,parent,props,type,inertia);
		if(contact)
			bn->createShapeNodeWith<VisualAspect,CollisionAspect,DynamicsAspect>(shape);
		else
			bn->createShapeNodeWith<VisualAspect, DynamicsAspect>(shape);

		bn->getShapeNodesWith<VisualAspect>().back()->getVisualAspect()->setColor(color);

		if(obj_file!="None" && create_obj)
		{
			std::string obj_path = std::string(MASS_ROOT_DIR)+"/data/OBJ/"+obj_file;
			const aiScene* scene = MeshShape::loadMesh(std::string(obj_path));

			MeshShapePtr visual_shape = std::shared_ptr<MeshShape>(new MeshShape(Eigen::Vector3d(0.01,0.01,0.01),scene));
			visual_shape->setColorMode(MeshShape::ColorMode::SHAPE_COLOR);
			auto vsn = bn->createShapeNodeWith<VisualAspect>(visual_shape);

			Eigen::Isometry3d T_obj;
			T_obj.setIdentity();			
			T_obj = T_body.inverse();
			vsn->setRelativeTransform(T_obj);
		}
	}

	std::cout<<"(DOFs : "<<skel->getNumDofs()<<")"<< std::endl;
	return skel;
}
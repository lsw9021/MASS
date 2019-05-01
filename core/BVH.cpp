#include "BVH.h"
#include <iostream>
#include <Eigen/Geometry>
#include "dart/dart.hpp"
using namespace dart::dynamics;
namespace MASS
{
Eigen::Matrix3d
R_x(double x)
{
	double cosa = cos(x*3.141592/180.0);
	double sina = sin(x*3.141592/180.0);
	Eigen::Matrix3d R;
	R<<	1,0		,0	  ,
		0,cosa	,-sina,
		0,sina	,cosa ;
	return R;
}
Eigen::Matrix3d R_y(double y)
{
	double cosa = cos(y*3.141592/180.0);
	double sina = sin(y*3.141592/180.0);
	Eigen::Matrix3d R;
	R <<cosa ,0,sina,
		0    ,1,   0,
		-sina,0,cosa;
	return R;	
}
Eigen::Matrix3d R_z(double z)
{
	double cosa = cos(z*3.141592/180.0);
	double sina = sin(z*3.141592/180.0);
	Eigen::Matrix3d R;
	R<<	cosa,-sina,0,
		sina,cosa ,0,
		0   ,0    ,1;
	return R;		
}
BVHNode::
BVHNode(const std::string& name,BVHNode* parent)
	:mParent(parent),mName(name),mChannelOffset(0),mNumChannels(0)
{

}
void
BVHNode::
SetChannel(int c_offset,std::vector<std::string>& c_name)
{
	mChannelOffset = c_offset;
	mNumChannels = c_name.size();
	for(const auto& cn : c_name)
		mChannel.push_back(CHANNEL_NAME[cn]);
}
void
BVHNode::
Set(const Eigen::VectorXd& m_t)
{
	mR.setIdentity();
	
	for(int i=0;i<mNumChannels;i++)
	{
		switch(mChannel[i])
		{
		case Xpos:break;
		case Ypos:break;
		case Zpos:break;
		case Xrot:mR = mR*R_x(m_t[mChannelOffset+i]);break;
		case Yrot:mR = mR*R_y(m_t[mChannelOffset+i]);break;
		case Zrot:mR = mR*R_z(m_t[mChannelOffset+i]);break;
		default:break;
		}
	}

}
void
BVHNode::
Set(const Eigen::Matrix3d& R_t)
{
	mR = R_t;
}
Eigen::Matrix3d
BVHNode::
Get()
{
	return mR;
}

void
BVHNode::
AddChild(BVHNode* child)
{
	mChildren.push_back(child);
}
BVHNode*
BVHNode::
GetNode(const std::string& name)
{
	if(!mName.compare(name))
		return this;

	for(auto& c : mChildren)
	{
		BVHNode* bn = c->GetNode(name);
		if(bn!=nullptr)
			return bn;
	}

	return nullptr;
}


BVH::
BVH(const dart::dynamics::SkeletonPtr& skel,const std::map<std::string,std::string>& bvh_map)
	:mSkeleton(skel),mBVHMap(bvh_map),mCyclic(true)
{

}

Eigen::VectorXd
BVH::
GetMotion(double t)
{
	int k = ((int)std::floor(t/mTimeStep));
	if(mCyclic)
		k %= mNumTotalFrames;
	k = std::max(0,std::min(k,mNumTotalFrames-1));
	double dt = t/mTimeStep - std::floor(t/mTimeStep);
	Eigen::VectorXd m_t = mMotions[k];
	
	for(auto& bn: mMap)
		bn.second->Set(m_t);
	
	
	int dof = mSkeleton->getNumDofs();
	Eigen::VectorXd p = Eigen::VectorXd::Zero(dof);

	for(auto ss : mBVHMap)
	{
		BodyNode* bn = mSkeleton->getBodyNode(ss.first);
		Eigen::Matrix3d R = this->Get(ss.second);
		Joint* jn = bn->getParentJoint();
		int idx = jn->getIndexInSkeleton(0);

		if(jn->getType()=="FreeJoint")
		{
			Eigen::Isometry3d T;
			T.translation() = 0.01*m_t.segment<3>(0);
			T.linear() = R;
			p.segment<6>(idx) = FreeJoint::convertToPositions(T);
		}
		else if(jn->getType()=="BallJoint")
			p.segment<3>(idx) = BallJoint::convertToPositions(R);
		else if(jn->getType()=="RevoluteJoint")
		{
			Eigen::Vector3d u =dynamic_cast<RevoluteJoint*>(jn)->getAxis();
			Eigen::Vector3d aa = BallJoint::convertToPositions(R);
			double val;
			if((u-Eigen::Vector3d::UnitX()).norm()<1E-4)
				val = aa[0];
			else if((u-Eigen::Vector3d::UnitY()).norm()<1E-4)
				val = aa[1];
			else
				val = aa[2];

			if(val>M_PI)
				val -= 2*M_PI;
			else if(val<-M_PI)
				val += 2*M_PI;

			p[idx] = val;
		}
	}

	return p;
}

Eigen::Matrix3d
BVH::
Get(const std::string& bvh_node)
{
	return mMap[bvh_node]->Get();
}
void
BVH::
Parse(const std::string& file,bool cyclic)
{
	mCyclic = cyclic;
	std::ifstream is(file);

	char buffer[256];

	if(!is)
	{
		std::cout<<"Can't Open File"<<std::endl;
		return;
	}
	while(is>>buffer)
	{
		if(!strcmp(buffer,"HIERARCHY"))
		{
			is>>buffer;//Root
			is>>buffer;//Name
			int c_offset = 0;
			mRoot = ReadHierarchy(nullptr,buffer,c_offset,is);
			mNumTotalChannels = c_offset;
		}
		else if(!strcmp(buffer,"MOTION"))
		{
			is>>buffer; //Frames:
			is>>buffer; //num_frames
			mNumTotalFrames = atoi(buffer);
			is>>buffer; //Frame
			is>>buffer; //Time:
			is>>buffer; //time step
			mTimeStep = atof(buffer);
			mMotions.resize(mNumTotalFrames);
			for(auto& m_t : mMotions)
				m_t = Eigen::VectorXd::Zero(mNumTotalChannels);
			double val;
			for(int i=0;i<mNumTotalFrames;i++)
			{
				for(int j=0;j<mNumTotalChannels;j++)
				{
					is>>val;
					mMotions[i][j]=val;
				}
			}
		}
	}
	is.close();

	BodyNode* root = mSkeleton->getRootBodyNode();
	std::string root_bvh_name = mBVHMap[root->getName()];
	Eigen::VectorXd m = mMotions[0];

	mMap[root_bvh_name]->Set(m);
	T0.linear() = this->Get(root_bvh_name);
	T0.translation() = 0.01*m.segment<3>(0);

	m = mMotions[mNumTotalFrames-1];

	mMap[root_bvh_name]->Set(m);
	T1.linear() = this->Get(root_bvh_name);
	T1.translation() = 0.01*m.segment<3>(0);


}
BVHNode*
BVH::
ReadHierarchy(BVHNode* parent,const std::string& name,int& channel_offset,std::ifstream& is)
{
	char buffer[256];
	double offset[3];
	std::vector<std::string> c_name;

	BVHNode* new_node = new BVHNode(name,parent);
	mMap.insert(std::make_pair(name,new_node));

	is>>buffer; //{

	while(is>>buffer)
	{
		if(!strcmp(buffer,"}"))
			break;
		if(!strcmp(buffer,"OFFSET"))
		{
			//Ignore
			double x,y,z;

			is>>x;
			is>>y;
			is>>z;
		}
		else if(!strcmp(buffer,"CHANNELS"))
		{

			is>>buffer;
			int n;
			n= atoi(buffer);
			
			for(int i=0;i<n;i++)
			{
				is>>buffer;
				c_name.push_back(std::string(buffer));
			}
			
			new_node->SetChannel(channel_offset,c_name);

			
			
			channel_offset+=n;
		}
		else if(!strcmp(buffer,"JOINT"))
		{
			is>>buffer;
			BVHNode* child = ReadHierarchy(new_node,std::string(buffer),channel_offset,is);
			new_node->AddChild(child);
		}
		else if(!strcmp(buffer,"End"))
		{
			is>>buffer;
			BVHNode* child = ReadHierarchy(new_node,std::string("EndEffector"),channel_offset,is);
			new_node->AddChild(child);
		}
	}
	
	return new_node;
}
std::map<std::string,MASS::BVHNode::CHANNEL> BVHNode::CHANNEL_NAME =
{
	{"Xposition",Xpos},
	{"XPOSITION",Xpos},
	{"Yposition",Ypos},
	{"YPOSITION",Ypos},
	{"Zposition",Zpos},
	{"ZPOSITION",Zpos},
	{"Xrotation",Xrot},
	{"XROTATION",Xrot},
	{"Yrotation",Yrot},
	{"YROTATION",Yrot},
	{"Zrotation",Zrot},
	{"ZROTATION",Zrot}
};
};
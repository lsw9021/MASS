#ifndef __MASS_MUSCLE_H__
#define __MASS_MUSCLE_H__
#include "dart/dart.hpp"

namespace MASS
{
struct Anchor
{
	int num_related_bodies;

	std::vector<dart::dynamics::BodyNode*> bodynodes;
	std::vector<Eigen::Vector3d> local_positions;
	std::vector<double> weights;

	Anchor(std::vector<dart::dynamics::BodyNode*> bns,std::vector<Eigen::Vector3d> lps,std::vector<double> ws);
	Eigen::Vector3d GetPoint();
};
class Muscle
{
public:
	Muscle(std::string _name,double f0,double lm0,double lt0,double pen_angle,double lmax);
	void AddAnchor(const dart::dynamics::SkeletonPtr& skel,dart::dynamics::BodyNode* bn,const Eigen::Vector3d& glob_pos,int num_related_bodies);
	void AddAnchor(dart::dynamics::BodyNode* bn,const Eigen::Vector3d& glob_pos);
	const std::vector<Anchor*>& GetAnchors(){return mAnchors;}
	void Update();
	void ApplyForceToBody();
	double GetForce();
	double Getf_A();
	double Getf_p();
	double Getl_mt();

	Eigen::MatrixXd GetJacobianTranspose();
	std::pair<Eigen::VectorXd,Eigen::VectorXd> GetForceJacobianAndPassive();

	int GetNumRelatedDofs(){return num_related_dofs;};
	Eigen::VectorXd GetRelatedJtA();

	std::vector<dart::dynamics::Joint*> GetRelatedJoints();
	std::vector<dart::dynamics::BodyNode*> GetRelatedBodyNodes();
	void ComputeJacobians();
	Eigen::VectorXd Getdl_dtheta();
public:
	std::string name;
	std::vector<Anchor*> mAnchors;
	int num_related_dofs;
	std::vector<int> related_dof_indices;

	std::vector<Eigen::Vector3d> mCachedAnchorPositions;
	std::vector<Eigen::MatrixXd> mCachedJs;
public:
	//Dynamics
	double g(double _l_m);
	double g_t(double e_t);
	double g_pl(double _l_m);
	double g_al(double _l_m);
	

	double l_mt,l_mt_max;
	double l_m;
	double activation;


	double f0;
	double l_mt0,l_m0,l_t0;

	double f_toe,e_toe,k_toe,k_lin,e_t0; //For g_t
	double k_pe,e_mo; //For g_pl
	double gamma; //For g_al
};

}
#endif
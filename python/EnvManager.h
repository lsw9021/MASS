#ifndef __ENV_MANAGER_H__
#define __ENV_MANAGER_H__
#include "Environment.h"
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <utility>
namespace py = pybind11;

class EnvManager
{
public:
	EnvManager(std::string meta_file,int num_envs);

	int GetNumState();
	int GetNumAction();
	int GetSimulationHz();
	int GetControlHz();
	int GetNumSteps();
	bool UseMuscle();

	void Step(int id);
	void Reset(bool RSI,int id);
	bool IsEndOfEpisode(int id);
	double GetReward(int id);

	void Steps(int num);
	void StepsAtOnce();
	void Resets(bool RSI);
	const Eigen::VectorXd& IsEndOfEpisodes();
	const Eigen::MatrixXd& GetStates();
	void SetActions(const Eigen::MatrixXd& actions);
	const Eigen::VectorXd& GetRewards();

	//For Muscle Transitions
	int GetNumTotalMuscleRelatedDofs(){return mEnvs[0]->GetNumTotalRelatedDofs();};
	int GetNumMuscles(){return mEnvs[0]->GetCharacter()->GetMuscles().size();}
	const Eigen::MatrixXd& GetMuscleTorques();
	const Eigen::MatrixXd& GetDesiredTorques();
	void SetActivationLevels(const Eigen::MatrixXd& activations);
	
	void ComputeMuscleTuples();
	const Eigen::MatrixXd& GetMuscleTuplesJtA();
	const Eigen::MatrixXd& GetMuscleTuplesTauDes();
	const Eigen::MatrixXd& GetMuscleTuplesL();
	const Eigen::MatrixXd& GetMuscleTuplesb();
private:
	std::vector<MASS::Environment*> mEnvs;

	int mNumEnvs;
	int muscle_torque_cols;
	int tau_des_cols;

	Eigen::VectorXd mEoe;
	Eigen::VectorXd mRewards;
	Eigen::MatrixXd mStates;
	Eigen::MatrixXd mMuscleTorques;
	Eigen::MatrixXd mDesiredTorques;

	Eigen::MatrixXd mMuscleTuplesJtA;
	Eigen::MatrixXd mMuscleTuplesTauDes;
	Eigen::MatrixXd mMuscleTuplesL;
	Eigen::MatrixXd mMuscleTuplesb;



};

#endif
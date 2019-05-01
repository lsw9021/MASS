#ifndef __NUMPY_HELPER_H__
#define __NUMPY_HELPER_H__
#include <vector>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
namespace p = boost::python;
namespace np = boost::python::numpy;
np::ndarray toNumPyArray(const std::vector<float>& val);
np::ndarray toNumPyArray(const std::vector<double>& val);
np::ndarray toNumPyArray(const std::vector<Eigen::VectorXd>& val);
np::ndarray toNumPyArray(const std::vector<Eigen::MatrixXd>& val);
np::ndarray toNumPyArray(const std::vector<std::vector<float>>& val);
np::ndarray toNumPyArray(const std::vector<std::vector<double>>& val);
np::ndarray toNumPyArray(const std::vector<bool>& val);
np::ndarray toNumPyArray(const Eigen::VectorXd& vec);
np::ndarray toNumPyArray(const Eigen::MatrixXd& matrix);
np::ndarray toNumPyArray(const Eigen::Isometry3d& T);
Eigen::VectorXd toEigenVector(const np::ndarray& array);
std::vector<Eigen::VectorXd> toEigenVectorVector(const np::ndarray& array);
Eigen::MatrixXd toEigenMatrix(const np::ndarray& array);
std::vector<bool> toStdVector(const p::list& list);
#endif
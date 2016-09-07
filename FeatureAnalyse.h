#pragma once
#include "globaltype.h"
#include <gcroot.h>
//declaration namespace
using namespace mlpack;
using namespace mlpack::kpca;
using namespace mlpack::kernel; 
using namespace mlpack::pca;
using namespace cli;

//! @brief The engine class for dimensional reduction or subspace transformation
/*  
*/
class FeatureAnalyse
{
public:
	FeatureAnalyse();
	~FeatureAnalyse();
	//void create(const std::string& detectorType);
	virtual void apply(std::string str) = 0;
	//virtual array<array<double>^>^ transform(array<array<double>^>^ arr);
	//virtual array<array<double>^>^ transform(array<array<double>^>^ arr,int dimension);
protected:
	arma::mat m_inputMat;
	int       m_dimensionNum;
	arma::mat m_transformedMat;
	arma::vec m_eigVal;
public:
	arma::mat getTransformationFeats(){return m_transformedMat;}
};

/** PCA subclass inherited from class FeatureAnalyse
 *
 */
class PCAFeatureAnalyse : public FeatureAnalyse
{
public:
	PCAFeatureAnalyse(arma::mat &data, int newDimension);
	PCAFeatureAnalyse(const arma::mat data);
	~PCAFeatureAnalyse();
	virtual void apply(std::string str);
private:
	PCA* m_pcaEngine;

};

class KPCAFeatureAnalyse : public FeatureAnalyse
{
public:
	KPCAFeatureAnalyse(arma::mat &data, int newDimension = 4);
	KPCAFeatureAnalyse(const arma::mat data);
	~KPCAFeatureAnalyse();
	virtual void apply(std::string str);
private:
	KernelPCA<GaussianKernel>* m_kpcaEngine;
};

public ref class KDAFeatureAnalyse
{
public:
	KDAFeatureAnalyse(array<array<double>^>^ arr,array<int>^ outputs,int dimension);
	~KDAFeatureAnalyse();
	array<array<double>^>^ transform(array<array<double>^>^ _arr,int dim);
	void compute();
	//array<array<double>^>^ transform(array<array<double>^>^ arr,int dimension);
private:
	KernelDiscriminantAnalysis^ m_kdaEngine;
	array<array<double>^>^ m_arr;
	array<int>^ m_output;
	int       m_dimensionNum;
};
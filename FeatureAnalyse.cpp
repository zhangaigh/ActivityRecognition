#include "FeatureAnalyse.h"

FeatureAnalyse::FeatureAnalyse()
{

}

FeatureAnalyse::~FeatureAnalyse()
{
}

/*
void FeatureAnalyse::create(const std::string& type)
{
	if (type.find("PCA") == 0 )
	{
		m_engine =  new PCA();
	}
	else if (type.find("KPCA") == 0)
	{
		m_engine = new KernelPCA<LinearKernel>;
	}
}*/

//Class PCAFeatureAnalyse implementation
PCAFeatureAnalyse::PCAFeatureAnalyse(const arma::mat _input)
{
	m_inputMat = _input;
	m_pcaEngine = new PCA();
}
PCAFeatureAnalyse::PCAFeatureAnalyse(arma::mat& _input,int _dim)
{
	m_inputMat = _input;
	m_dimensionNum = _dim;
	m_pcaEngine = new PCA();
}
PCAFeatureAnalyse::~PCAFeatureAnalyse()
{
	delete m_pcaEngine;
}
/*
PCA维数降解数据矩阵行为数据维数，列为数据点数
*/
void PCAFeatureAnalyse::apply(std::string str)
{
	arma::mat _mat = trans(m_inputMat);
	if (str.find("dr") == 0)            //dimension reduction
	{
		
		m_pcaEngine->Apply(_mat,  m_dimensionNum);
		m_transformedMat = _mat;
	}
	else if (str.find("tf") == 0)      //space transformation
	{
		m_pcaEngine->Apply(_mat, m_transformedMat, m_eigVal);
	}
	
}

//Class KPCAFeatureAnalyse implementation
KPCAFeatureAnalyse::KPCAFeatureAnalyse(const arma::mat _input) 
{
	m_inputMat = _input;
	m_kpcaEngine = new KernelPCA<GaussianKernel>;
}
KPCAFeatureAnalyse::KPCAFeatureAnalyse(arma::mat& _input,int _dim)
{
	m_inputMat = _input;
	m_dimensionNum = _dim;
	m_kpcaEngine = new KernelPCA<GaussianKernel>;
}
KPCAFeatureAnalyse::~KPCAFeatureAnalyse()
{
	delete m_kpcaEngine;
}

void KPCAFeatureAnalyse::apply(std::string str)
{
	arma::mat _mat = trans(m_inputMat);
	if (str.find("dr") == 0)            //dimension reduction
	{
		m_kpcaEngine->Apply(_mat,  m_dimensionNum);
		m_transformedMat = _mat;
	}
	else if (str.find("tf") == 0)      //space transformation
	{
		m_kpcaEngine->Apply(_mat, m_transformedMat, m_eigVal);
	}

}



KDAFeatureAnalyse::KDAFeatureAnalyse(array<array<double>^>^ _arr,array<int>^ outputs,int _dim)
{
	m_arr          = _arr;
	m_dimensionNum = _dim;
	m_output       = outputs;
	m_kdaEngine = gcnew KernelDiscriminantAnalysis(_arr, outputs, gcnew Gaussian());
		// the threshold is used to keep all components
	m_kdaEngine->Threshold = 0.01;
	m_kdaEngine->Regularization = 0.0001;

}
KDAFeatureAnalyse::~KDAFeatureAnalyse()
{

}

array<array<double>^>^ KDAFeatureAnalyse::transform(array<array<double>^>^ _arr,int dim)
{
	 
	array<array<double>^>^ tra = m_kdaEngine->Transform(_arr);
	return tra;
}

void KDAFeatureAnalyse::compute()
{
	m_kdaEngine->Compute();
}
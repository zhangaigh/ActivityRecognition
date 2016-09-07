#include "FeatureExtractor.h"

FeatureExtractor::FeatureExtractor()
{
	m_rawStream.resize(ACC_AXIS_NUM,SLIDE_WINDOW_WIDTH);
}
FeatureExtractor::~FeatureExtractor()
{
	
}

void FeatureExtractor::extract(int index)
{
	switch (index)
	{
#ifdef _FEATURE_MEAN_
	case featureType::MEAN:
		extractMEAN();
		break;
#endif //_FEATURE_MEAN_
#ifdef _FEATURE_VAR_
	case featureType::VAR:
		extractVAR();
		break;
#endif //_FEATURE_VAR_
#ifdef _FEATURE_AAD_
	case featureType::AAD:
		extractAAD();
		break;
#endif //_FEATURE_AAD_
#ifdef _FEATURE_SQRT_
	case featureType::SQRT:
		extractSQRT();
		break;
#endif //_FEATURE_SQRT_
#ifdef _FEATURE_PBD_
	case featureType::PBD:
		extractPBD();
		break;
#endif //_FEATURE_PBD_
#ifdef _FEATURE_AV_
	case featureType::AV:
		break;
#endif //_FEATURE_AV_
#ifdef _FEATURE_SMA_
	case featureType::SMA:
		extractSMA();
		break;
#endif //_FEATURE_SMA_
	case featureType::CORR:
		extractCorr();
	default:
		break;

	}
}

#ifdef _FEATURE_MEAN_
void FeatureExtractor::extractMEAN()
{
	arma::vec _vec = arma::mean(m_rawStream,1);
	m_extractedFeatures.insert_rows(m_extractedFeatures.n_rows,_vec);
}
#endif //_FEATURE_MEAN_

#ifdef _FEATURE_VAR_
void FeatureExtractor::extractVAR()
{
	arma::vec _vec = arma::var(m_rawStream,0,1);
	m_extractedFeatures.insert_rows(m_extractedFeatures.n_rows,_vec);
}
#endif //_FEATURE_VAR

#ifdef _FEATURE_AAD_
void FeatureExtractor::extractAAD()
{
	// needed to be verified. Dimension 3
	arma::vec _mean = arma::mean(m_rawStream,1);
	arma::mat _mat  = m_rawStream;
	_mat.each_col() -= _mean;
	m_extractedFeatures.insert_rows(m_extractedFeatures.n_rows,arma::mean(arma::abs(_mat),1));

}
#endif

#ifdef _FEATURE_SQRT_
void FeatureExtractor::extractSQRT()
{
	// Dimension 1
	arma::vec _vec = arma::mean(sqrt(sum(arma::square(m_rawStream),0)),1);
	m_extractedFeatures.insert_rows(m_extractedFeatures.n_rows,_vec);

}
#endif //_FEATURE_SQRT_

#ifdef _FEATURE_PBD_
void FeatureExtractor::extractPBD()
{
	umat _umat = hist(trans(m_rawStream),6);
	mat _mat(_umat.n_rows,_umat.n_cols);
	for (UINT32 j = 0; j < _mat.n_rows; j ++)
	{
		for (UINT32 k = 0; k < _mat.n_cols; k ++)
		{
			_mat(j,k) = _umat(j,k);
		}
	}
	for (UINT32 i = 0; i < _mat.n_cols; i++)
	{
		arma::vec _vec = _mat.col(i);

		m_extractedFeatures.insert_rows(m_extractedFeatures.n_rows,_vec);
	}
}
#endif

#ifdef _FEATURE_SMA_
void FeatureExtractor::extractSMA()
{
	arma::vec _vec = arma::mean(arma::sum(arma::abs(m_rawStream),0),1);
	m_extractedFeatures.insert_rows(m_extractedFeatures.n_rows,_vec);
}
#endif

#ifdef _FEATURE_AV_
void FeatureExtractor::extractAV()
{

}
#endif

void FeatureExtractor::extractAR()
{
	long double mean;
	std::vector<long double> params, sigma2e, gain, autocor;
	for (int j = 0; j < m_rawStream.n_rows; j ++)
	{
		stdvec vec = conv_to< stdvec >::from(m_rawStream.row(j));
		ar::burg_method(vec.begin(), vec.end(), mean, g_uiMaxOrder,
		back_inserter(params), back_inserter(sigma2e),
		back_inserter(gain), back_inserter(autocor),
		false,
		true);
		//best = ar::evaluate_models<AIC>(m_rawStream.n_cols, 0u, sigma2e.begin(), sigma2e.end());
		std::vector<double> tempVec;
		tempVec.assign( params.end()-g_uiMaxOrder,params.end());
		arma::vec _vec = conv_to< arma::vec >::from(tempVec);
		m_extractedFeatures.insert_rows(m_extractedFeatures.n_rows,_vec);
	}
	
}

void FeatureExtractor::extractCorr()
{
	arma::vec _mean = arma::mean(m_rawStream,1);
	arma::vec _std      = arma::stddev(m_rawStream,0,1);
	arma::mat _mat  = m_rawStream;
	_mat.each_col() -= _mean;
	_mat.each_col() /= _std;
	int n = SLIDE_WINDOW_WIDTH -1;
	double xy = (arma::dot(_mat.row(0),_mat.row(1)))/n;
	double xz = arma::dot(_mat.row(0),_mat.row(2))/n;
	double yz = arma::dot(_mat.row(1),_mat.row(2))/n;
	arma::vec _vec;
	_vec << xy << xz <<yz<<arma::endr;
	m_extractedFeatures.insert_rows(m_extractedFeatures.n_rows,_vec);
}


void FeatureExtractor::setRawStream(const arma::mat _mat)
{
	m_rawStream.copy_size(_mat);
	m_rawStream = _mat;
}
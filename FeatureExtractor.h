
#ifndef _FEATUREDESCRIPTOR_H_
#define _FEATUREDESCRIPTOR_H_
#include <vector>
#include <iostream>
#include <string>

#include "globaltype.h"
#include <numeric>
//#include <flann/flann.hpp>
//#include <Eigen/Dense>
#include "armadillo"

#include "ar.hpp"

//using namespace Eigen;
using namespace arma;
	//! @brief The class for computing diverse features for classification.
	/*    The class could be used to provide diverse feature computation for training or classifying activity data 
	*/
	class FeatureExtractor
	{
	public:
		 //! @brief A constructor.
		 /* A more elaborate description of the constructor.
		 */
		FeatureExtractor();

		 //! @brief A destructor.
		 /* A more elaborate description of the destructor.
		 */
		~FeatureExtractor();

		 //! @brief A member function for feature extraction.
		 /* The function could be used to extract corresponding feature .
		  \param a an integer argument which is the index of feature type
		  \return void
		 */
		void extract(int);

		/** return extracted feature vector
		 *
		 * @return vec 
		 */
		inline arma::vec getExtractedFeatures(){return m_extractedFeatures;}

		//inline void setRawStream(const arma::mat _mat){m_rawStream = _mat;}
		void setRawStream(const arma::mat _mat);
	private:
		arma::vec m_extractedFeatures;   /**< extracted feature vector */
		arma::mat m_rawStream;           /**< raw accelerator data */
	private:
		/** A member function for extracting mean feature 
		 */
#ifdef _FEATURE_MEAN_
		void extractMEAN();
#endif //_FEATURE_MEAN_
#ifdef _FEATURE_VAR_
		void extractVAR();
#endif //_FEATURE_VAR_
#ifdef _FEATURE_AAD_
		void extractAAD();
#endif //_FEATURE_AAD_
#ifdef _FEATURE_SQRT_
		void extractSQRT();
#endif //_FEATURE_SQRT_
#ifdef _FEATURE_PBD_
		void extractPBD();
#endif //_FEATURE_PBD_
#ifdef _FEATURE_SMA_
		void extractSMA();
#endif //_FEATURE_SMA_
#ifdef _FEATURE_AV_
		void extractAV();
#endif //_FEATURE_AV_
		/**  extract autoregressive model 
		 *
		 *  extract autoregressive model coefficients via open source Autoregressive process modeling tools.
		 *  
		 *  See the current <a
		 *  href="https://github.com/RhysU/ar/blob/master/README.rst"> README</a> for a
		 * more detailed overview and http://github.com/RhysU/ar for project
		 * information.
		 */
		void extractAR();

		/**  
		 *    extract the correlation feature among three accelerate axises
		 *   
		 */
		void extractCorr();
	};

#endif
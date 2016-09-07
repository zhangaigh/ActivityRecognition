#ifndef _DATAPROC_H_
#define _DATAPROC_H_
#include "globaltype.h"
//#include <Eigen/Dense>
using namespace arma;
//using namespace Eigen;

extern void mat2array(const arma::mat _mat,cli::array<array<double>^>^ _arr);
extern void vec2array(const arma::uvec _vec,cli::array<int>^ _arr);
extern void array2mat(cli::array<array<double>^>^ _arr,arma::mat& _mat);
	//! @brief The class for operation related preprocessing.
	/*  
	*/
	class DataProc
	{
	public:

		struct parameters 
		{  
			UINT32 uiSlideWinWidth;
			UINT32 uiSlideWinStep;
		};

		DataProc(parameters parma);
		DataProc();
		~DataProc();
	//! @brief moving average filter for accelerator signal
	/*  Enable the function using macro _FILTER_MOVING_SMOOTHING_
	*/
#ifdef _FILTER_MOVING_SMOOTHING_
		void filterMV(const MatrixXd& dataArray);
#endif //_FILTER_MOVING_SMOOTHING_
    //! @brief low pass filter for accelerator signal
	/* Enable the function using macro _FILTER_LOW_PASS_
	*/
#ifdef _FILTER_LOW_PASS_
		void filterLP();
#endif //_FILTER_LOW_PASS_

	//! get the data cube segmented from raw features
	/*  
	*/
	inline arma::cube getSegmentedAcceleraterStream(){return m_cSegmentedDataSlices;}

	/**
	* segment accelerator data 
	*/
	BOOL8 segment();

	/**
	* prepare data for preprocessing
	*/
	inline void setDataMatirx(const arma::mat _mat){m_mRawDataStream = _mat;}

	/**
	* concatenate extracted features into matrix for further dimension reduction processing
	* @param _extractedFeature the feature vector
	*/
	void concatExtractedFeatureToMat(arma::vec _extractedFeature);

	/**
	*  get the extracted feature matrix
	*/
	//inline arma::mat getExtractedFeatures(){return m_mExtractedFeature;}
	arma::mat getExtractedFeatures();

   /**
    *  compute feature scale and save the corresponding mean and std
    *  @ param _mat the input matrix for computing scale
    *  @param SAVE the flag for saving or not saving scale
    */
	static void computeFeatureScale(arma::mat &_mat ,bool SAVE = true,em_featurescale_type _type = EM_FEATURESCALE_STANDARDSCALER);
	static std::string int2String(int _type);
	private:
		void normalization();

	private:
		arma::cube    m_cSegmentedDataSlices;   ///for storing final segemented data with 3*m_uiWinWidth*N
		arma::mat     m_mRawDataStream;
		UINT32        m_uiWinWidth;
		UINT32        m_uiWinStep;

		arma::mat     m_mExtractedFeature;      ///for storing extracted feature matrix (column-wise )lo

		arma::vec              m_mean;                                 // feature mean for feature scale
		arma::vec              m_std;                                      //std for feature scale 



};

#endif //_DATAPROC_H_

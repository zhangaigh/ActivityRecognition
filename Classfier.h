//////////////////////////////////////////////////////////////////////////
/* COPYRIGHT NOTICE
 * Copyright (c) 2013 Shanghai advanced research institute
 * All rights reserved.
*/ 
//////////////////////////////////////////////////////////////////////////

/// @file 
/// @brief The SVM classifier class  
/// 
/// The svm classifier which wraps accord.net framework .
/// 
/// @version 1.0
/// @author bob
/// @date 09/30/2013

#pragma once
#include "globaltype.h"
#include "DataProc.h"
#include "floatfann.h"
#include "fann_cpp.h"
#include "ActivityRecognition.h"
using namespace cli;

/** The abstract classifier class
 *
 */
public ref class Classifier abstract         
{
public:
	Classifier(){};
	~Classifier(){};
	virtual void learning() = 0;
	//virtual std::vector<int> classify(array<array<double>^>^ inputs) = 0;   //@bob
	virtual std::vector<int> classify(arma::mat& inputs) = 0;
	virtual int  classify(array<double>^ _arr)  = 0;
	
	virtual void saveModelToFile(std::string _str) = 0;
	virtual void loadModelFromFile(std::string _str) = 0;

	virtual void setTrainData(array<array<double>^>^, array<int>^) = 0;

protected:
	INT32 m_iNumOfClass;

};  // Classifier end

/** Inherited SVM classifier class. 
 *  
 *  A subclass for implementing SVM classify algorithm
 */
public ref class SVMClassifier : public Classifier
{
public:
	//SVMClassifier(array<array<double>^>^ inputs,array<int>^ outputs,int numOfClass);
	SVMClassifier(int length,int numOfClass);
	~SVMClassifier(){};
	virtual void learning() override;
	//virtual std::vector<int> classify(array<array<double>^>^ inputs) override;  @bob
	virtual std::vector<int> classify(arma::mat& inputs) override;
	virtual int  classify(array<double>^ _arr)  override;

	virtual void saveModelToFile(std::string _str) override;
	virtual void loadModelFromFile(std::string _str) override;

	virtual void setTrainData(array<array<double>^>^, array<int>^) override;
private:
	MulticlassSupportVectorMachine^ m_ksvm;
	MulticlassSupportVectorLearning^ m_ml;
	int                             m_numOfClass;
	array<array<double>^>^          m_inputs;
	array<int>^                     m_outputs;
};  // SVMClassifier end


/** Struct for neural network parameters
	 *
	 *\note The struct is defined here because it can't be define in a managed class.
	 */
	struct SNeuralNetworkParameter
	{
		
		UINT32      uiNumOfLayers;   /**< number of layers */
		UINT32      uiNumOfInput;    /**< the number of neurons in input layer   */
		UINT32      uiNumOfHidden;      /**< the number of neurons in hidden layer   */
		UINT32      uiMumOfOutput;     /**< the number of neurons in output layer */
		FLOAT32     fDesiredError;     /**< the desired error crition */
		UINT32      uiNumOfMaxIter;    /**< the maximum iteration number */
		FLOAT32     fLearingRate;      /**< learning rate */
		 
		SNeuralNetworkParameter()
		{
			
		}
	};

/** The neural network class 
 *
 *	The class is used to implement neural network framework.
 *	\note This class is the wrapper of FANN lib. For more information please refer to <a href="http://leenissen.dk/fann/wp/">Fast Artificial Neural Network Library</a> online.
 */

class CNeuralNetwork 
{
public:

	/** Class constructor
	 *
	 */
	CNeuralNetwork(int numOfClass);
	CNeuralNetwork();
	~CNeuralNetwork();
	/** learning function for neural network
	 *
	 *  train the nn via some predefined param and train data 
	 **/
	virtual void learning();


	virtual int classify(FLOAT32* _data);
	/** Set neural network parameters
	 *
	 *  @param param the parameters needed by nn
	 *  \note refer to struct SNeuralNetworkParameter for parameters list
	 *  \sa SNeuralNetworkParameter
	 */ 
	void setNeuralNetworkParm(CActivityRecognition::SNeuralNetworkParameter param);

	/** Set data source for train 
	 *
	 * The data is obtained from HDF5 file. Furthermore, the data is further transformed into reduced dimension via DR method such as PCA\KPCA\KDA.
	 * The data matrix represent train data set. The number of row denotes feature dimension and the column number of matrix is the number of train sample.
	 * The sample label is a column vector in which each entry is the label
	 * @param _trainMat reduced train data 
	 * @param _trainLabel the corresponding sample label 
	 */
	void setTrainData(arma::mat _trainMat, arma::uvec _trainLabel);

	void saveModelToFile(const std::string _str);
	void loadModelFromFile(const std::string _str);


private:
	FANN::neural_net *m_nnEngine;
	FANN::training_data m_nnTrainData;
private:

	UINT32      m_uiNumOfLayers;   /**< number of layers */
	UINT32      m_iNumOfClass;     /**< number of class */
	UINT32      m_uiNumOfInput;    /**< the number of neurons in input layer   */
	UINT32      m_uiNumOfHidden;      /**< the number of neurons in hidden layer   */
	UINT32      m_uiMumOfOutput;     /**< the number of neurons in output layer */
	FLOAT32     m_fDesiredError;     /**< the desired error crition */
	UINT32      m_uiNumOfMaxIter;    /**< the maximum iteration number */
	FLOAT32     m_fLearingRate;      /**< learning rate */

};
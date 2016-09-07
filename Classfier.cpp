#include "Classfier.h"

ISupportVectorMachineLearning^ delSVM(KernelSupportVectorMachine^ svm, array<array<double>^>^ inputs, array<int>^ outputs, int class1, int class2)
{
	SequentialMinimalOptimization^ smo = gcnew SequentialMinimalOptimization(svm, inputs, outputs);
	smo->Epsilon = 0.001;
	smo->Complexity = 1.0000;
	smo->Tolerance  = 0.2000;
	return smo;
}

/*
SVMClassifier::SVMClassifier(array<array<double>^>^ inputs,array<int>^ outputs,int numOfClass)
{
	m_ksvm = gcnew MulticlassSupportVectorMachine(inputs[0]->Length, gcnew Gaussian(11.0), numOfClass);
	//m_ml = gcnew MulticlassSupportVectorLearning(m_ksvm, inputs, outputs);
	//m_inputs = inputs;
} */

SVMClassifier::SVMClassifier(int length,int numOfClass)
{
	//m_ksvm = gcnew MulticlassSupportVectorMachine(inputs[0]->Length, gcnew Gaussian(11.0), numOfClass);
	//m_ml = gcnew MulticlassSupportVectorLearning(m_ksvm, inputs, outputs);
	//m_inputs = inputs;
	m_ksvm = gcnew MulticlassSupportVectorMachine(length, gcnew Gaussian(11.0), numOfClass);
}

void SVMClassifier::learning()
{
	m_ml = gcnew MulticlassSupportVectorLearning(m_ksvm, m_inputs, m_outputs);
	m_ml->Algorithm = gcnew SupportVectorMachineLearningConfigurationFunction(delSVM);
	m_ml->Run();
}

// classify data matrix
//std::vector<int> SVMClassifier::classify(array<array<double>^>^ inputs)    @bob
std::vector<int> SVMClassifier::classify(arma::mat &  inputs)
{
	//convert mat to array
	array<array<double>^>^ _arr = gcnew array<array<double>^>(inputs.n_rows);
	for (int i = 0; i < inputs.n_rows;i ++)
	{
		_arr[i] = gcnew array<double>(inputs.n_cols);
	}
	mat2array(inputs,_arr);


	std::vector<int> output;
	//for (int i = 0; i < inputs->Length; i ++) //bob
	for (int i = 0; i < _arr->Length; i ++)
	{
		//int results = m_ksvm->Compute(inputs[i]); @bob
		int results = m_ksvm->Compute(_arr[i]);
		output.push_back(results);
		//std::cout<<"classify output:"<<results<<std::endl;
	}
	return output;
}

// classify one test vector
int SVMClassifier::classify(array<double>^ _arr)
{
	return m_ksvm->Compute(_arr);
}

void SVMClassifier::saveModelToFile(std::string _str)
{
	m_ksvm->Save( gcnew System::String(_str.c_str()));
}
void SVMClassifier::loadModelFromFile(std::string _str)
{
	m_ksvm = MulticlassSupportVectorMachine::Load(gcnew System::String(_str.c_str()));
	//m_ksvm->Load(gcnew System::String(_str.c_str()));

}

void SVMClassifier::setTrainData(array<array<double>^>^ _input, array<int>^ _output)
{
	m_inputs = _input;
	m_outputs = _output;
}

/************************************************************************/
/*  Implementation for class CNeuralNetwork                                                                    */
/************************************************************************/
CNeuralNetwork::CNeuralNetwork()
{
	
}
CNeuralNetwork::CNeuralNetwork(int numOfClass)
{
	m_nnEngine = new FANN::neural_net();
	m_iNumOfClass = numOfClass;
}

CNeuralNetwork::~CNeuralNetwork()
{
	delete m_nnEngine;
}

void CNeuralNetwork::setNeuralNetworkParm(CActivityRecognition::SNeuralNetworkParameter _param)
{
	UINT32* layers = new UINT32(_param.uiNumOfLayers);
	m_nnEngine->create_standard_array(_param.uiNumOfLayers,layers);
	m_nnEngine->set_learning_rate(_param.fLearingRate);
	m_nnEngine->set_activation_steepness_hidden(1.0);
	m_nnEngine->set_activation_steepness_output(1.0);
	m_nnEngine->set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC_STEPWISE);
	m_nnEngine->set_activation_function_output(FANN::SIGMOID_SYMMETRIC_STEPWISE);
	m_nnEngine->set_training_algorithm(FANN::TRAIN_QUICKPROP);

	m_uiNumOfMaxIter = _param.uiNumOfMaxIter;
	m_fDesiredError  = _param.fDesiredError;
	delete[] layers;
}

void CNeuralNetwork::learning()
{
	m_nnEngine->init_weights(m_nnTrainData);
	m_nnEngine->train_on_data(m_nnTrainData,m_uiNumOfMaxIter,0,m_fDesiredError);
}

int CNeuralNetwork::classify(FLOAT32* _data)
{
	FLOAT32* output = new FLOAT32[m_iNumOfClass];
	output = m_nnEngine->run(_data);
	for (int i = 0; i < m_iNumOfClass; i ++)
	{
		if (1 == output[i])
		{
			return i+1;
		}
	}
	return 0;
}

//std::vector<int> CNeuralNetwork::classify(array<array<double>^>^ inputs)
//{
//
//}
//} 

void CNeuralNetwork::setTrainData(arma::mat _trainMat, arma::uvec _trainLabel)
{
	DOUBLE* p = _trainMat.memptr();
	UINT32  uiNumOfCols = _trainMat.n_cols;
	UINT32  uiNumOfRows = _trainMat.n_rows;

	//allocate input pointer array for fann port
	FLOAT32 **input;
	input = new FLOAT32*[uiNumOfCols];
	for (int i = 0; i <  uiNumOfCols; i ++)
	{
		input[i] = new FLOAT32[uiNumOfRows];
		input[i] = (FLOAT32*)_trainMat.colptr(i);
	}

	/**allocate output pointer array
	*  the format is as following:
	*
	*/
	FLOAT32 **output;
	output = new FLOAT32*[uiNumOfCols];
	for (int k = 0; k < uiNumOfCols; k ++)
	{
		output[k] = new FLOAT32[m_iNumOfClass];
		memset(output[k],0,m_iNumOfClass);
		int ind = _trainLabel[uiNumOfCols];
		output[k][ind-1] = 1;
	}

	//set train data via invoking fann lib port
	m_nnTrainData.set_train_data(uiNumOfCols, 
								 uiNumOfRows,input,
								 m_iNumOfClass,output);

	//delete allocated memory
	for(int j=0;j < uiNumOfCols;j++)
	{
		delete []input[j];
	}
	delete []input;
}

void CNeuralNetwork::saveModelToFile(const std::string _str)
{
	m_nnEngine->save_to_fixed(_str);
}

void CNeuralNetwork::loadModelFromFile(const std::string _str)
{
	m_nnEngine->create_from_file(_str);
}
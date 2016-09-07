#include "DataReader.h"
#include "FeatureExtractor.h"
#include "globaltype.h"
#include "IoHDF5.h"
#include "FeatureAnalyse.h"
#include "DataProc.h"
#include "Classfier.h"
#include "ActivityRecognition.h"

CActivityRecognition::CActivityRecognition()
{
	
}

CActivityRecognition::~CActivityRecognition()
{
}

/**
*  The implementation for training classifier and save the train result into hdf5 format.
*  Below is the main steps:
*  1) read all data using readAllData()
*  2) extract raw features through class FeatureExtractor
*  3) transform raw features into another dimension though dimension reduction method (PCA\KPCA\KDA)
*  Below is the directory architecture of training sets
*  /root
*       /walking
*              /walking_yymmdd.txt
*       /running
*       /walking
*       /standing
*       ......
*/

void CActivityRecognition::train()
{

	if (m_sClassifierType.compare("SVM") == 0)
	{
		classifierEngine->learning();
	}
	else if (m_sClassifierType.compare("NN") == 0)
	{
		
		NNEngine->learning();
	}
}

/**
* The implementation for classifying the data into different activities from on line data. Blow is the main steps:
* 1) read online data using readOnlineData()
* 2) extract raw features through class FeatureExtractor
* 3) transform raw features into another dimension though dimension reduction method (PCA\KPCA\KDA)
* 4) apply classification algorithms to transformed feature (KDA\flann\svm)
*/

int* CActivityRecognition::classify(const std::string& _str,int& num)
{
	CIoHDF5 IoHdf5;
	//std::string _baseDir = "..\\Datasets\\test";
	//IoHdf5.loadDataFromHDF5( _baseDir + "\\" + _str);
	IoHdf5.loadDataFromHDF5(m_basepath + "\\test\\" + _str);
	arma::mat _mat    = IoHdf5.getTrainMatrix();

	//feature scale 
	_mat = trans(_mat);
	DataProc::computeFeatureScale(_mat,false,EM_FEATURESCALE_MINMAXSCALER);

	// dimension reduction
	int* ptr;
	_mat = trans(_mat);
	//array<array<double>^>^ _arr = dimensitionReduction(_mat);
	array<array<double>^>^ _arr = process(_mat); 

	arma::mat dr_mat;     //@bob
	array2mat(_arr,dr_mat);   //@bob

	if (m_sClassifierType.find("SVM") == 0)
	{
		//classify the data
		std::vector<int> vec =  classifierEngine->classify(dr_mat);
		num = vec.size();
		ptr = new int[vec.size()];
		memcpy(ptr,&vec[0],vec.size()*sizeof(int));
		return ptr;
	}
	else if (m_sClassifierType.find("NN") == 0)
	{

	}
}



void CActivityRecognition::loadTrainData(const std::string str)
{
	// load train data from hdf5 file
	CIoHDF5 IoHdf5;
	//std::string _str(m_basepath+);
		//std::string __str = _str + str;
	IoHdf5.loadDataFromHDF5(  m_basepath  + "\\train\\"  + str);

	//IoHdf5.loadDataFromHDF5( m_basepath + "\\" + str);
	m_mTrainMat    = IoHdf5.getTrainMatrix();
	m_vTrainLabels = IoHdf5.getTrainLabel();
	
	m_mTrainMat = trans(m_mTrainMat);
	DataProc::computeFeatureScale(m_mTrainMat,true,EM_FEATURESCALE_MINMAXSCALER);
	//dimension reduction
	m_mTrainMat = trans(m_mTrainMat);


}

static bool CActivityRecognition::createData(const std::string& _fileStr)
{
	CDataReader datareader;
	DataProc::parameters param;
	param.uiSlideWinWidth = m_uiWinWidth;
	param.uiSlideWinStep  = m_uiWinStep;
	std::string str(m_basepath +"\\test\\" + _fileStr);
	CIoHDF5 IoHdf5(str);
	if (datareader.readDir( m_basepath + "\\test" ))
	{
		std::vector<std::string> dirlist  = datareader.getDirsList();
		std::vector<std::string> filelist = datareader.getFilesList(); 
		for (int i = 0; i < dirlist.size();i ++)
		{
			std::string str(dirlist[i]);
			dirlist[i] =  m_basepath + "\\test\\" + dirlist[i];
			IoHdf5.createPosture(str);
			DataProc    dataproc(param);
			if (datareader.readDir(dirlist[i]))
			{
				filelist = datareader.getFilesList();

				if (!filelist.empty())
				{
					for (int j = 0; j < filelist.size(); j ++)
					{
						filelist[j] = dirlist[i] + "\\" + filelist[j];
						if (datareader.readFile(filelist[j]))
						{
							arma::mat _mat = datareader.getFileData();
							dataproc.setDataMatirx(_mat);
							if (dataproc.segment())
							{
								arma::cube stream = dataproc.getSegmentedAcceleraterStream();
								for (UINT32 i = 0; i < stream.n_slices; i ++)
								{
									// 	Todo: 
									FeatureExtractor featureEngine;
									featureEngine.setRawStream(stream.slice(i));
									featureEngine.extract(0);
									featureEngine.extract(1);
									featureEngine.extract(2);
									featureEngine.extract(3);
									featureEngine.extract(4);
									featureEngine.extract(5);
									featureEngine.extract(7);
									//featureEngine.getExtractedFeatures().print("Extracted Feature:");
									dataproc.concatExtractedFeatureToMat(featureEngine.getExtractedFeatures());
								}
							}
							else
							{
								continue;
							}
						}
					}  //end for 
				} //end readdir
			}  // end for 
			
		     IoHdf5.writeFeaturesLabelsToHDF5(str,dataproc.getExtractedFeatures());   ///write posture data into hdf5 file
		}
	
		return 1;

	} 
	else
	{
		return 0;
	}
}

static bool CActivityRecognition::createTrainData(const std::string& _fileStr)
{
	CDataReader datareader;
	DataProc::parameters param;
	param.uiSlideWinWidth = m_uiWinWidth;
	param.uiSlideWinStep  = m_uiWinStep;
	//std::string str( m_basepath +"\\" + _fileStr);
	std::string str( m_basepath +"\\train\\" + _fileStr);
	CIoHDF5 IoHdf5(str);
	
	if (datareader.readDir(m_basepath + "\\train"))
	{
		std::vector<std::string> dirlist  = datareader.getDirsList();
		std::vector<std::string> filelist = datareader.getFilesList(); 
		for (int i = 0; i < dirlist.size();i ++)
		{
			std::string str(dirlist[i]);
			dirlist[i] = m_basepath + "\\train\\" + dirlist[i];
			IoHdf5.createPosture(str);
			DataProc    dataproc(param);
			if (datareader.readDir(dirlist[i]))
			{
				filelist = datareader.getFilesList();

				if (!filelist.empty())
				{
					for (int j = 0; j < filelist.size(); j ++)
					{
						filelist[j] = dirlist[i] + "\\" + filelist[j];
						if (datareader.readFile(filelist[j]))
						{
							arma::mat _mat = datareader.getFileData();
							dataproc.setDataMatirx(_mat);
							if (dataproc.segment())
							{
								arma::cube stream = dataproc.getSegmentedAcceleraterStream();
								for (UINT32 i = 0; i < stream.n_slices; i ++)
								{
									// 	Todo: 
									FeatureExtractor featureEngine;
									featureEngine.setRawStream(stream.slice(i));
									featureEngine.extract(0);
									featureEngine.extract(1);
									featureEngine.extract(2);
									featureEngine.extract(3);
									//featureEngine.extract(4);
									featureEngine.extract(5);
									featureEngine.extract(7);
									//featureEngine.getExtractedFeatures().print("Extracted Feature:");
									dataproc.concatExtractedFeatureToMat(featureEngine.getExtractedFeatures());
								}
							}
							else
							{
								continue;
							}
						}
					}  //end for 
				} //end readdir
			}  // end for 

			IoHdf5.writeFeaturesLabelsToHDF5(str,dataproc.getExtractedFeatures());   ///write posture data into hdf5 file
		}
		return 1;

	} 
	else
	{
		return 0;
	}
}
// create train data from UCI Har dataset
unsigned int CActivityRecognition::createTrainData(const std::string& _file, const std::vector<int> _index)
{
	//create class label
	arma::mat labelMat;
	std::vector<arma::uvec> vecIndex;
	if (mlpack::data::Load(m_basepath + "\\train\\" + "subject_train.csv",labelMat,false))
	{
		arma::vec _vec = trans(labelMat.row(0));
		for (int i = 0; i < _index.size(); i ++)
		{
			arma::uvec _uvec = arma::find(_vec == _index[i] );
			vecIndex.push_back(_uvec);
		}
		
		//IoHdf5.writeFeaturesLabelsToHDF5(_vec,dataproc.getExtractedFeatures());  
	}
	else
	{
		mlpack::Log::Fatal<< "can't read label file!" << std::endl;
		return -1;
	}
	 
	  //create class data
	   CDataReader datareader;
		DataProc::parameters param;
		param.uiSlideWinWidth = m_uiWinWidth;
		param.uiSlideWinStep  = m_uiWinStep;
		std::string str( m_basepath +"\\train\\" + _file);
		CIoHDF5 IoHdf5(str);
		if (datareader.readDir(m_basepath + "\\train"))
		{
			std::vector<std::string> dirlist  = datareader.getDirsList();
			//DataProc    dataproc(param);
			for (int i = 0; i < dirlist.size();i ++)
			{
				dirlist[i] = m_basepath + "\\train\\" + dirlist[i];
				// IoHdf5.createPosture(str);
				if (datareader.readDir(dirlist[i]))
				{
					std::vector<std::string> filelist = datareader.getFilesList();
					if (!filelist.empty())
					{
						std::vector<arma::mat> accVector;
						for (int j = 0; j < filelist.size(); j ++)
						{
							filelist[j] = dirlist[i] + "\\" + filelist[j];
							arma::mat tempMat;
							 if (mlpack::data::Load(filelist[j],tempMat,false))
							 {
							        //accStream.insert_slices(accStream.n_slices,tempMat);
									accVector.push_back(tempMat);
							 }
							 else
							 {
								 mlpack::Log::Fatal<< "can't read file!" << std::endl;
							 }
						}
						for (int ind = 0; ind < _index.size(); ind ++)
						{
							DataProc    dataproc(param);
							std::string str;
							if (_index[ind] == 1)   // walking
							{
								str = "walking";
								IoHdf5.createPosture(str);
							}
							else if (_index[ind] == 5)   //standing
							{
								str = "standing";
								IoHdf5.createPosture(str);
							}
							else if (_index[ind] == 6) //lying
							{
								str = "lying";
								IoHdf5.createPosture(str);
							}
							
							arma::mat __mat = accVector[0].cols(vecIndex[ind]);
							int indexCols = __mat.n_cols;
							//int indexCols = accVector[0].cols(vecIndex[ind]).n_cols;
							for (UINT32 i = 0; i < indexCols; i ++)
							{
								FeatureExtractor featureEngine;
								arma::mat fragment;
								for (int j = 0; j < filelist.size(); j ++)
								{
									arma::mat classMat = accVector[j].cols(vecIndex[ind]);
									arma::mat _vec = classMat.col(i);
									fragment.insert_rows(fragment.n_rows,trans(_vec));
									//fragment.insert_rows(trans(accVector[j].cols(vecIndex[ind]).col(i)));
								}
								featureEngine.setRawStream(fragment);
								featureEngine.extract(0);
								featureEngine.extract(1);
								featureEngine.extract(2);
								featureEngine.extract(3);
								featureEngine.extract(4);
								featureEngine.extract(5);
								dataproc.concatExtractedFeatureToMat(featureEngine.getExtractedFeatures());
								
							}   //end for 
							IoHdf5.writeFeaturesLabelsToHDF5(str,dataproc.getExtractedFeatures());
					}  //end for special class
				} //end read dir
			}  // end for 
		}
			
	}

		return 0;
	
}

//设置分类器类型，支持三种分类方法：SVM、NN、层级分类方法
//如果默认设置，则代表层级分类方法
void CActivityRecognition::setClassifier(const std::string _str)
{
	m_sClassifierType = _str;
	if (_str.compare("SVM") == 0)
	{
		classifierEngine = gcnew SVMClassifier(m_uiDataLength,m_uiNumOfType);
		return;
		
	}
	else if(_str.compare("NN") == 0)
	{
		NNEngine        = new CNeuralNetwork(m_uiNumOfType);
		return;
	}

	classifierEngine = gcnew SVMClassifier(m_uiDataLength,m_uiNumOfType);
	NNEngine        = new CNeuralNetwork(m_uiNumOfType);
}

void CActivityRecognition::setARParam(_sParameters param)
{
	m_basepath    =  param.sBasePath;
	m_uiNumOfType =  param.uiNumOfType;
	m_uiWinWidth  =  param.uiWinWidth;
	m_uiWinStep   =  param.uiWinStep;
	m_sDrType     =  param.sDrType;
	m_sClassifierType =  param.sClassifierType;
	m_uiDataLength = param.uiDataLength;

	g_sDrType     = param.sDrType;
	g_numOfType   = param.uiNumOfType;

	g_uiMaxOrder  = param.uiMaxOrder;

}


void CActivityRecognition::saveModelToFile(const std::string _str,const std::string _type)
{
	if (_type.find("NN") == 0)
	{
		NNEngine->saveModelToFile(_str);
	}
	else
	{
		classifierEngine->saveModelToFile(_str);
	}
	
}


void CActivityRecognition::loadModelFromFile(const std::string _str,const std::string _type)
{
	if (_type.find("NN")==0)
	{
		NNEngine->loadModelFromFile(_str);
	}
	else
	{
		classifierEngine->loadModelFromFile(_str);
	}
}

void CActivityRecognition::setTrainData()
{
	array<int>^ outputs = gcnew array<int>(m_vTrainLabels.n_rows);
	vec2array(m_vTrainLabels,outputs);
	array<array<double>^>^ inputs_raw = gcnew array<array<double>^>(m_mTrainMat.n_rows);
	for (int i = 0; i < m_mTrainMat.n_rows;i ++)
	{
		inputs_raw[i] = gcnew array<double>(m_mTrainMat.n_cols);
	}
	mat2array(m_mTrainMat,inputs_raw);

	array<array<double>^>^ inputs = dimensitionReduction(m_mTrainMat);
	

	if (m_sClassifierType.compare("NN") == 0)
	{
		arma::mat input_mat;
		array2mat(inputs,input_mat);
		NNEngine->setTrainData(input_mat,m_vTrainLabels);
	}
	else
	{
		classifierEngine->setTrainData(inputs,outputs);
	}
}

// 返回值：
// -1 无识别状态 0 walking 1 running 2 sitting
int CActivityRecognition::classify(float accX,float accY,float accZ)
{
	arma::vec _vec;
	_vec<<accX<<accY<<accZ<<arma::endr;
		m_mAccData.insert_cols(m_mAccData.n_cols,_vec);
		if (m_mAccData.n_cols == m_uiWinWidth)
		{
			//todo 特征提取、维数降解、分类
			// 特征提取
			FeatureExtractor featureEngine;
			featureEngine.setRawStream(m_mAccData);
			featureEngine.extract(0);
			featureEngine.extract(1);
			featureEngine.extract(2);
			featureEngine.extract(3);
			//featureEngine.extract(4);
			featureEngine.extract(5);
			featureEngine.extract(7);
			arma::vec featureVec = featureEngine.getExtractedFeatures();
			//featureVec =(featureVec - arma::min(featureVec)) /(arma::max(featureVec)-arma::min(featureVec));		
			arma::mat _mat;
			_mat.insert_cols(0,featureVec);
			//feature scale
			DataProc::computeFeatureScale(_mat,false,EM_FEATURESCALE_MINMAXSCALER);
			//dimension reduction
			//array<array<double>^>^ feature =  dimensitionReduction(trans(_mat));
			array<array<double>^>^ feature = process(trans(_mat)); 
			//slide windows
			m_mAccData(span::all,span(0,m_uiWinStep-1)) = m_mAccData(span::all,span(m_uiWinStep,m_uiWinWidth-1));
			m_mAccData.shed_cols(m_uiWinStep,m_uiWinWidth-1);
			//classify the data 
			return classifierEngine->classify(feature[0]);
		}
		else
		{
				return -1;
		}
}

int CActivityRecognition::classify(std::vector<float>_dataList)
{
	int num = _dataList.size();
	//std::cout<<" data length:>>>>"<<num<<std::endl;
	if (_dataList.empty())
	{
		return -1;
	}
	else
	{

		arma::colvec _vec = arma::conv_to<colvec>::from(_dataList);
		m_mAccData.set_size(3,num/3);
		std::memcpy(m_mAccData.memptr(),_vec.memptr(),num * sizeof(double));
			
		// feature extraction
		FeatureExtractor featureEngine;
		featureEngine.setRawStream(m_mAccData);
		featureEngine.extract(0);
		featureEngine.extract(1);
		featureEngine.extract(2);
		featureEngine.extract(3);
		//featureEngine.extract(4);
		featureEngine.extract(5);
		featureEngine.extract(7);
		arma::vec featureVec = featureEngine.getExtractedFeatures();
		//featureVec.print();
		//feature scale
		arma::mat _mat;
		_mat.insert_cols(0,featureVec);
		DataProc::computeFeatureScale(_mat,false,EM_FEATURESCALE_MINMAXSCALER);
		//dimension reduction
		array<array<double>^>^ feature = process(trans(_mat)); 


		m_mAccData.set_size(0,0);
		//classify data 
		return classifierEngine->classify(feature[0]);
	}
}

void CActivityRecognition::initPreProcess()
{
	// process dimension reduction 
	arma::mat _mat = m_mTrainMat;
	arma::mat _transformedMat;
	if (g_sDrType.find("PCA") == 0)
	{
		PCAFeatureAnalyse _pca(_mat,g_numOfType-1);
		_pca.apply("dr");
		_transformedMat = _pca.getTransformationFeats();
	}
	else if (g_sDrType.find("KPCA") == 0)
	{
		KPCAFeatureAnalyse _kpca(_mat,g_numOfType-1);
		_kpca.apply("dr");
		_transformedMat = _kpca.getTransformationFeats();
	}
	else if (g_sDrType.find("KDA") == 0)
	{
		array<array<double>^>^ inputs = gcnew array<array<double>^>(m_mTrainMat.n_rows);
		for (int i = 0; i < m_mTrainMat.n_rows;i ++)
		{
			inputs[i] = gcnew array<double>(m_mTrainMat.n_cols);
		}
		mat2array(m_mTrainMat,inputs);
		array<int>^ outputs = gcnew array<int>(m_vTrainLabels.n_rows);
		vec2array(m_vTrainLabels,outputs);
		 kdaEngine = gcnew KDAFeatureAnalyse(inputs,outputs,g_numOfType);
		 kdaEngine->compute();
	}
}

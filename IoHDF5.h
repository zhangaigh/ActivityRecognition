#pragma once

#include "H5Cpp.h"
#include "globaltype.h"
using namespace H5;


typedef struct {
    int index;               // index of the current object 
} iter_info;

const UINT32 RANK = 2;
extern arma::mat g_TrainMatrix;
extern arma::uvec g_TrainLabel;

/**
*   Class related to file Input/output using HDF5 format
*/
class CIoHDF5
{
public:

	typedef struct _sPosture
	{
		Group       group;
		DataSet     data; 
		DataSet     label;
		postureType type;
	}sData;

	/**
	*  define get_all_groups as friend function  
	*/
	//friend herr_t get_all_groups(hid_t loc_id, const char *name, void *opdata);
	CIoHDF5();
	CIoHDF5(const std::string str);
	~CIoHDF5();
	/** 
	*  create special posture hdf5 group,dataset in hfFile
	*  @param str the name of special posture type (walking,running,sitting,lying)
	*/
	void createPosture(const std::string _str);

		/** write according features and label into hdf5 file using batch label and feature file
	 * @param _vec the activity label vector
	 * @param _mat the extracted feature matrix
	 **/
	void writeFeaturesLabelsToHDF5(const std::string _str,const arma::mat _mat);

	/**
	*  @brief load hdf5 dataset into arma::mat
	*  The method is mainly to load training data for activity recognition
	*  @param _file the file name
	*  @param _mat  the matrix for storing data
	*  @param _vec  the label 
    */
	void loadDataFromHDF5(const std::string _file);

	inline arma::mat getTrainMatrix(){return trans(g_TrainMatrix);}
	inline arma::uvec getTrainLabel(){return g_TrainLabel;}

private:
	H5File                             m_h5File;               /// pointer for h5 file handle 

	std::map<std::string,_sPosture>     m_mapData;               /// record the whole data 
    
};



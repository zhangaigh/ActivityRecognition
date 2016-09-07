#include "IoHDF5.h"

arma::mat g_TrainMatrix;
arma::uvec g_TrainLabel;
postureType stringToEnum(const std::string _str)
{
	if (_str == "walking")
	{
		return walking;
	}else if (_str == "running")
	{
		return running;
	}else if (_str == "sitting")
	{
		return sitting;
	}else if (_str == "lying")
	{
		return lying;
	}else if (_str == "standing")
	{
		return standing;
	}
}

UINT32 EnumToINT(postureType type)
{
	/*
	if (type == walking)
	{
		return 0;
	}
	else if (type == running)
	{
		return 1;
	} 
	else if (type == sitting)
	{
		return 2;
	}
	else if (type == lying)
	{
		return 3;
	}
	else if (type == standing)
	{
		return 4;
	}
	*/
	int index = 0;
	switch (type)
	{
	case lying:
		 index = 0;
		 break;
	case standing:
		index = 1;
		break;
	case walking:
		index = 2;
		break;
	case running:
		index = 3;
		break;
	default:
		break;
	}
	return index;
}

//operation function for iterating over hdf5 group 
herr_t get_groups(hid_t loc_id, const char *name, void *opdata)
{
	iter_info *info=(iter_info *)opdata;
	Group group = H5Gopen2(loc_id, name, H5P_DEFAULT);
	//read "data" dataset
	DataSet dataset = group.openDataSet("data");
	DataSpace dataspace = dataset.getSpace();
	hsize_t dims_out[2];
	UINT32 ndims = dataspace.getSimpleExtentDims( dims_out, NULL);
	if (dims_out[0]!=0 && dims_out[1]!=0)
	{
		//arma::mat      tempMat(dims_out[0],dims_out[1]); 
		arma::mat      tempMat(dims_out[1],dims_out[0]); 

		//arma::mat _temMat = trans(tempMat);
		dataset.read(tempMat.memptr(),PredType::NATIVE_DOUBLE);

		//tempMat.print("tempMat:");
		arma::mat      _mat = trans(tempMat);
		//_mat.print("_mat");
		UINT32 cols = g_TrainMatrix.n_cols;
		UINT32 rows = g_TrainMatrix.n_rows;
		g_TrainMatrix.resize(dims_out[0],cols+dims_out[1]);
		g_TrainMatrix.submat(arma::span::all,arma::span(cols,cols+dims_out[1]-1)) = _mat;
		//g_TrainMatrix.print("g_TrainMatrix");
		//read "label" dataset
		DataSet label = group.openDataSet("label");
		dataspace = label.getSpace();
		ndims = dataspace.getSimpleExtentDims( dims_out, NULL);
		arma::uvec      tempVec(dims_out[0]);
		label.read(tempVec.memptr(),PredType::NATIVE_INT);
		//cols = g_TrainLabel.n_cols;
		rows = g_TrainLabel.n_rows;
		g_TrainLabel.resize(rows+dims_out[0]);
		g_TrainLabel.subvec(rows,rows+dims_out[0]-1) = tempVec;
		(info->index)++;
	}
	return 0;
 }


CIoHDF5::CIoHDF5(const std::string str)// :
			// m_h5File(H5File(str,H5F_ACC_TRUNC))
{
	H5File file(str.c_str(),H5F_ACC_TRUNC);
	m_h5File = file;
}
CIoHDF5::CIoHDF5()
{
}
CIoHDF5::~CIoHDF5()
{
	//delete m_h5File;
}
void CIoHDF5::createPosture(const std::string _str)
{
	// create group from root 
	std::string str = "/" + _str;
	//create group
	Group group = m_h5File.createGroup(str.c_str());

	//Create the data space with unlimited dimensions.

	hsize_t      dims1[2]    ={0,0};  // dataset dimensions at creation
	hsize_t      maxdims1[2] = {H5S_UNLIMITED, H5S_UNLIMITED};
	DataSpace mspace1( RANK, dims1,maxdims1);

	hsize_t      dims2[1]    = {0};  // dataset dimensions at creation
	hsize_t      maxdims2[1] = {H5S_UNLIMITED};
	DataSpace mspace2( 1, dims2,maxdims2);
	//create datatype
	DSetCreatPropList cparms1;
	hsize_t      chunk_dims[2] ={16, 16};
	cparms1.setChunk( RANK, chunk_dims );
	DSetCreatPropList cparms2;
	hsize_t      chunk_dims1[1] ={3};
	cparms2.setChunk( 1, chunk_dims1 );
	DataSet data  = group.createDataSet("data",PredType::NATIVE_DOUBLE, mspace1,cparms1);
	DataSet label = group.createDataSet("label",PredType::NATIVE_INT, mspace2,cparms2);

	sData sDataStruct = {group,data,label,stringToEnum(_str)};

	m_mapData.insert(make_pair(_str,sDataStruct));
}

void CIoHDF5::writeFeaturesLabelsToHDF5(const std::string _str,const arma::mat _mat)
{
	Group group = m_h5File.openGroup(_str.c_str());
	DataSet data = group.openDataSet("data");
	
	///write data into hdf5//// 
	DataSpace fspace1 = data.getSpace();
	hsize_t dims_out[2];
	UINT32 ndims = fspace1.getSimpleExtentDims( dims_out, NULL);
	hsize_t      size[2];
	//size[0]   = dims_out[0];
	size[0]   = _mat.n_rows;
	size[1]   = dims_out[1] + _mat.n_cols;
	data.extend( size );
	hsize_t     offset[2];
	DataSpace fspace2 = data.getSpace ();
	offset[0] = 0;
	offset[1] = dims_out[1];
	hsize_t dims[2]  = {_mat.n_rows,_mat.n_cols};
	DataSpace mspace2( RANK, dims );
	fspace2.selectHyperslab( H5S_SELECT_SET, dims, offset );
	//write data into dataset
	arma::mat t_mat = trans(_mat);
	data.write(t_mat.memptr(), PredType::NATIVE_DOUBLE, mspace2, fspace2 );

	////write label data///////
	DataSet label = group.openDataSet("label");
	DataSpace label_space = label.getSpace();
	hsize_t dims_out1[2];
	ndims = label_space.getSimpleExtentDims( dims_out1, NULL);
	hsize_t      size1[1];
	size1[0]   = dims_out1[0] + _mat.n_cols;
	//size1[1]   = dims_out1[1];
	label.extend( size1 );
	hsize_t     offset_label[1];
	DataSpace label_space1 = label.getSpace ();
	offset_label[0] = dims_out1[0];
	//offset_label[1] = 0;
	hsize_t dims1[1]  = {_mat.n_cols};
	DataSpace mspace3( 1, dims1 );
	label_space1.selectHyperslab( H5S_SELECT_SET, dims1, offset_label );
	
	arma::uvec labelVec(_mat.n_cols);
	labelVec.fill(EnumToINT(stringToEnum(_str)));
	label.write(labelVec.memptr(),PredType::NATIVE_INT,mspace3,label_space1);
}

void CIoHDF5::loadDataFromHDF5(const std::string _file)
{
	//open the hdf5 file
	H5File file;
	file.openFile(_file.c_str(),H5F_ACC_RDONLY);
	iter_info info;
	int idx = 0;
	info.index = 0;
	g_TrainMatrix.reset();
	g_TrainLabel.reset();
	// iterate all over all the groups
	idx = file.iterateElems("/", NULL, get_groups, &info);
}

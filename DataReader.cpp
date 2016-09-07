#include "DataReader.h"

CDataReader::CDataReader()
{
	m_vFileNames.clear();
	m_vDirNames.clear();
}

BOOL8 CDataReader::readDir(std::string dirName)
{
	if (!m_vDirNames.empty())
	{
		m_vDirNames.clear();
	}
	if (!m_vFileNames.empty())
	{
		m_vFileNames.clear();
	}
	if (!m_mFileData.is_empty())
	{
		m_mFileData.reset();
	}
	//open a directory the WIN32 way
	HANDLE hFind = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATAA fdata;        

	if(dirName[dirName.size()-1] == '\\' || dirName[dirName.size()-1] == '/') 
	{
		dirName = dirName.substr(0,dirName.size()-1);
	}

	hFind = FindFirstFileA(std::string(dirName).append("\\*").c_str(), &fdata);	 
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			if (std::strcmp(fdata.cFileName, ".") != 0 &&
				std::strcmp(fdata.cFileName, "..") != 0)
			{
				if (fdata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
				{
					m_vDirNames.push_back(fdata.cFileName);
				}
				else
				{
					m_vFileNames.push_back(fdata.cFileName);
				}
			}
		}
		while (FindNextFileA(hFind, &fdata) != 0);
	} 
	else 
	{
		mlpack::Log::Fatal << "can't open directory!" << std::endl;
		return false;
	}
	FindClose(hFind);
	return true;
}

BOOL8 CDataReader::readFile(const std::string file)
{
	if (!m_vFileNames.empty())
	{
		m_vFileNames.clear();
	}
	if (!m_mFileData.is_empty())
	{
		m_mFileData.reset();
	}
	if (mlpack::data::Load(file,m_mFileData,false))
	{
		removeRepeatedData();
		
		//mlpack::data::Save(file,m_mFileData);
		return true;
	}
	else
	{
		mlpack::Log::Fatal << "can't read file!" << std::endl;
		return false;
	}
}

void CDataReader::removeRepeatedData()
{
	arma::mat tempMat;
	arma::mat transMat = trans(m_mFileData);
	//int i = 0;
	int start = 0;
	while (start< (transMat.n_rows-1))
	{	
		while (compareVec(transMat.row(start),transMat.row(start+1)) )
		{
			 transMat.shed_row(start+1);
			//  i ++;
			 if (start == (transMat.n_rows -1))
			 {
				 break;
			 }
		
		}
		//tempMat.insert(tempMat.n_rows,m_mFileData.row(start) );
		start ++;

	//	i = start;
	}
	m_mFileData.clear();
	m_mFileData = trans(transMat);
	/*for (int i = 0; i < m_mFileData.n_rows; i ++)
	{
		if (m_mFileData.row(i) == m_mFileData.row(i+1))
		{

		}
	}*/
}

int CDataReader::compareVec(arma::rowvec _vec1,arma::rowvec _vec2)
{
//	int result = 1;
	for (int i = 0; i < _vec1.n_cols; i ++)
	{
		if (_vec1(i) != _vec2(i))
		{
	//		result = 0;
			return 0;
		}
	}
	return 1;
}
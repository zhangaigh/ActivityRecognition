#ifndef _GLOBALTYPE_H_
#define  _GLOBALTYPE_H_

//data type definition
#include "TypeDef.h"

//some include files related to mlpack lib
#include <mlpack/core.hpp>
#include <mlpack/methods/pca/pca.hpp>
#include <mlpack/methods/kernel_pca/kernel_pca.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>

#include "armadillo"

//define test macro switch
#ifndef _TEST_ARMARDILLIO_
#define _TEST_ARMARDILLIO_
#endif //_TEST_ARMARDILLIO_

#ifndef _TEST_FLANNAR_
#define _TEST_FLANNAR_
#endif

//some namespace related to Accord.net framework. Meanwhile 
/*
#using "D:\\Program Files\\Accord.NET\\Framework\\Release\\Accord.dll"
#using "D:\\Program Files\\Accord.NET\\Framework\\Release\\Accord.MachineLearning.dll"
#using "D:\\Program Files\\Accord.NET\\Framework\\Release\\Accord.Statistics.dll"
#using "D:\\Program Files\\Accord.NET\\Framework\\Release\\Accord.Math.dll"*/

using namespace cli;
using namespace Accord;
using namespace Accord::MachineLearning::VectorMachines;
using namespace Accord::MachineLearning::VectorMachines::Learning;
using namespace Accord::Math;
using namespace Accord::Statistics;
using namespace Accord::Statistics::Kernels;
using namespace Accord::Statistics::Analysis;


#define SLIDE_WINDOW_WIDTH 512
#define ACC_AXIS_NUM       3
#define SLIDE_WINDOW_STEP  256

/** An enum type.
* The enum data structure for storing feature type
*/
enum featureType 
{
	MEAN,                       /**< mean */
	VAR,                        /**< stand deviation */ 
	AAD,                        /**< average absolute difference */
	SQRT,                       /**< average resultant acceleration */
	PBD,                        /**< probabilistic binned distribution */
	SMA,                        /**< Signal magnitude areas */
	AV,                          /**< Accumulated variety */
	CORR                      /**<Correlation between axis> */
};

enum postureType
{
	walking,
	running,
	sitting,
	lying,
	standing
};

enum em_featurescale_type
{
	EM_FEATURESCALE_MINMAXSCALER,
	EM_FEATURESCALE_STANDARDSCALER
};


#ifndef _FEATURE_MEAN_
#define _FEATURE_MEAN_
#endif   

#ifndef _FEATURE_VAR_
#define _FEATURE_VAR_
#endif 

#ifndef _FEATURE_AAD_
#define _FEATURE_AAD_
#endif 

#ifndef _FEATURE_SQRT_
#define _FEATURE_SQRT_
#endif 

#ifndef _FEATURE_PBD_
#define _FEATURE_PBD_
#endif 

#ifndef _FEATURE_SMA_
#define _FEATURE_SMA_
#endif 

#ifndef _FEATURE_AV_
#define _FEATURE_AV_
#endif 

#ifndef _FILTER_MOVING_SMOOTHING_
//#define _FILTER_MOVING_SMOOTHING_
#endif //_FILTER_MOVING_SMOOTHING_

#ifndef _FILTER_LOW_PASS_
#define _FILTER_LOW_PASS_
#endif //_FITLER_LOW_PASS_

/** Global variable for storing maximum order of autoregressive model
 */
extern UINT32 g_uiMaxOrder; 

#endif // _GLOBALTYPE_H_
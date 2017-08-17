#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <assert.h>
#include <chrono>
#include <cstring>
#include <iomanip>

#include "TimeSeries.h"

using namespace std;

#define DOUBLE_MAX_VALUE 1e20

static double lb_kim_prune_rate_sum = 0;
static int total_1nn_call = 0;
static double lb_kim_prune_rate_current_test = 0;
static int total_1nn_call_current_test = 0;

std::string datasets[] = {
	 "50words",
	"Adiac",
	"ArrowHead",
	"Beef",
	"BeetleFly",
	"BirdChicken",
	"Car",
	"CBF",
	"ChlorineConcentration",
	"CinC_ECG_torso",
	"Coffee",
	"Computers",
	"Cricket_X",
	"Cricket_Y",
	"Cricket_Z",
	"DiatomSizeReduction",
	"DistalPhalanxOutlineAgeGroup",
	"DistalPhalanxOutlineCorrect",
	"DistalPhalanxTW",
	"Earthquakes",
	"ECG200",
	"ECG5000",
	"ECGFiveDays",
	"ElectricDevices",
	"FaceAll",
	"FaceFour",
	"FacesUCR",
	"FISH",
	"FordA",
	"FordB",
	"Gun_Point",
	"Ham",
	"HandOutlines",
	"Haptics",
	"Herring",
	"InlineSkate",
	"InsectWingbeatSound",
	"ItalyPowerDemand",
	"LargeKitchenAppliances",
	"Lighting2",
	"Lighting7",
	"MALLAT",
	"Meat",
	"MedicalImages",
	"MiddlePhalanxOutlineAgeGroup",
	"MiddlePhalanxOutlineCorrect",
	"MiddlePhalanxTW",
	"MoteStrain",
	"NonInvasiveFatalECG_Thorax1",
	"NonInvasiveFatalECG_Thorax2",
	"OliveOil",
	"OSULeaf",
	"PhalangesOutlinesCorrect",
	"Phoneme",
	"Plane",
	"ProximalPhalanxOutlineAgeGroup",
	"ProximalPhalanxOutlineCorrect",
	"ProximalPhalanxTW",
	"RefrigerationDevices",
	"ScreenType",
	"ShapeletSim",
	"ShapesAll",
	"SmallKitchenAppliances",
	"SonyAIBORobotSurface",
	"SonyAIBORobotSurfaceII",
	"StarLightCurves",
	"Strawberry",
	"SwedishLeaf",
	"Symbols",
	"synthetic_control",
	"ToeSegmentation1",
	"ToeSegmentation2",
	"Trace",
	"Two_Patterns",
	"TwoLeadECG",
	"uWaveGestureLibrary_X",
	"uWaveGestureLibrary_Y",
	"uWaveGestureLibrary_Z",
	"UWaveGestureLibraryAll",
	"wafer",
	"Wine",
	"WordsSynonyms",
	"Worms",
	"WormsTwoClass",
	"yoga"
};


/*
 * Finds the best AA transformed DTW distance for the test case.
 * Both test and train_set must be already having their Jpoints set with the same s
 */
int classify_1nn(const AA::TimeSerie& test, const AA::SharedTimeSerieDataset& train_set){
	double bsf = DOUBLE_MAX_VALUE;
	int label = 0;
	double kim = 0;
	for(std::size_t i = 0 ; i < train_set.size() ; i++){
#ifdef KIMH		
		double lb_kim = AA::lb_kim_hierarchy(test, *train_set[i], bsf);
#elif KIMMINMAX		
		double lb_kim = AA::lb_kim_minmax(test, *train_set[i], bsf);
#else		
		double lb_kim = 0;
#endif
		if(lb_kim < bsf){
			double tmp_dist = AA::distance(test, *train_set[i], bsf);
			if(tmp_dist < bsf){
				bsf = tmp_dist;
				label = train_set[i]->label();
				//std::cerr << "new bsf" << bsf << std::endl;
			}
		}else{
			kim++;
		}
	}
	lb_kim_prune_rate_sum += kim/train_set.size();

	lb_kim_prune_rate_current_test += kim/train_set.size();
	total_1nn_call_current_test++;
	total_1nn_call++;
	return label;
}


/**
 * Validate the accuracy of the 1NN for the given test and train set.
 * Assumes both have been already AA transformed with equal S.
 */
double validate( const AA::SharedTimeSerieDataset& train_set, const AA::SharedTimeSerieDataset& test_set){
	double accuracy = 0;
	double w_mean, w_sdev;
	AA::normalizeWeightOnDataset(train_set,&w_mean, &w_sdev);
	for(std::size_t i = 0 ; i < test_set.size() ; i ++ ){
		test_set[i]->normalizeWeight(w_mean,w_sdev);
		if(classify_1nn(*test_set[i],train_set) == test_set[i]->label()){
			accuracy++;
		}
	}
	return accuracy / test_set.size();	
}


void load_dataset(const char *filename, AA::SharedTimeSerieDataset& dataset){
	std::cerr << "Loading file " << filename << std::endl;
	ifstream file(filename);
	std::shared_ptr<AA::TimeSerie> TS = make_shared<AA::TimeSerie>();
	int n = 0;
	while(file >> *TS){
		dataset.push_back(TS);
		TS = make_shared<AA::TimeSerie>();
		n++;
	}
	std::cerr << " Read " << n << " Timeseries into dataset" << std::endl;
	file.close();
}

/*
 * Performs K-fold cross validation and returns the best value of s in ss.
 * The dataset most probabely shuffled after call
 */
double cross_validate( AA::SharedTimeSerieDataset& dataset, int K,  const std::vector<double>& Ss){	
	std::vector<AA::SharedTimeSerieDataset> folds(K);
	int fold_size = dataset.size() / K;
	std::cerr << "-------------- " << K << "-fold CV ... -------------"	<< std::endl;
	double best_s_so_far = Ss[0];
	double best_acc_so_far = 0;
	for(double s : Ss){
		std::cerr << "##### s = " << s << std::endl;
		for(int i = 0 ; i < dataset.size() ; i++){
			dataset[i]->setStatesAndJpoints(s, 3);
		}
		int n = dataset.size(); // total folds sizes covered so far 
		double acc_sum = 0;
		while(n > 0){
			AA::SharedTimeSerieDataset::iterator next_fold_end;
			if(n >= 2 * fold_size){
				next_fold_end = std::begin(dataset) + fold_size;
			}else{
				next_fold_end = std::begin(dataset) + n;
			}
			AA::SharedTimeSerieDataset holdout{	std::begin(dataset) , next_fold_end};
			dataset.erase(	std::begin(dataset) , next_fold_end);
			double acc = validate(dataset, holdout);
			std::copy(holdout.begin(), holdout.end(), 
						std::back_inserter(dataset));	// push holdset back
			n -= holdout.size();
			acc_sum += acc;
		//	std::cerr << "n is "<< n << " acc was " << acc << std::endl;	
		}		
		assert(n == 0);
		double tmp_acc = acc_sum / K;
		std::cerr << "------------------------------------------" << std::endl;
		fprintf(stderr, "... s = %2f -> \t accuracy = %3f, error = %3f\n", s , tmp_acc, 
				1- tmp_acc);
		std::cerr << "------------------------------------------" << std::endl;
		if(tmp_acc > best_acc_so_far){
			best_acc_so_far = tmp_acc;
			best_s_so_far = s;
			if (best_acc_so_far == 1) break;
		}
	}

	fprintf(stderr, "------------ Best S = %2f with Best ACCURACY = %3f,\t ERROR = %3f\n",
								best_s_so_far, best_acc_so_far, 1 - best_acc_so_far);
	return best_s_so_far;
}


void cv_test_all(std::vector<std::string> datasets, int kfold, const std::vector<double> Ss){

	std::chrono::time_point<std::chrono::system_clock> startTime = 
			std::chrono::system_clock::now();
	for(int i = 0 ; i < datasets.size(); i++){
		std::cout << "======================================" << std::endl;
		std::cout << "Testing " << datasets[i] << " ... " << std::endl;
		std::cout << "======================================" << std::endl;
		std::chrono::time_point<std::chrono::system_clock> thisStartTime = 
			std::chrono::system_clock::now();
		AA::SharedTimeSerieDataset train_dset;
		std::string train_path = datasets[i] + "/" + datasets[i] + "_TRAIN";
		load_dataset(train_path.c_str(), train_dset);
		AA::SharedTimeSerieDataset test_dset;
		std::string test_path = datasets[i] + "/" + datasets[i] + "_TEST";
		load_dataset(test_path.c_str(), test_dset);
		
		std::cout << "Train loading time: \t" 
				<< std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - thisStartTime).count() / 1000.0
				<< " s" << std::endl;
		thisStartTime = std::chrono::system_clock::now();
		double best_s = cross_validate(train_dset, kfold, Ss);
		std::cout << "Training time (CV): \t" 
				<< std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - thisStartTime).count() / 1000.0
				<< " s" << std::endl;
		thisStartTime = std::chrono::system_clock::now();
		// Retransfom train and test with the best s
		for(int j = 0 ; j < train_dset.size() ; j++){
			train_dset[j]->setStatesAndJpoints(best_s, 3);
			//std::cout << *train_dset[j];
		}
		double train_w_mean, train_w_sdev;
		AA::normalizeWeightOnDataset(train_dset, &train_w_mean, &train_w_sdev);
		for(int j = 0 ; j < test_dset.size() ; j++){
			test_dset[j]->setStatesAndJpoints(best_s, 3);
			test_dset[j]->normalizeWeight(train_w_mean, train_w_sdev);
			//std::cout << test_dset[j];
		}
		double acc = validate(train_dset,test_dset);
		std::cout << "Testing time: \t" 
				<< std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - thisStartTime).count() / 1000.0
				<< " s" << std::endl;	
		std::cout <<  " TEST ACCURACY = " << acc << "\t ERROR = " << 1 - acc << std::endl;
		std::cout  << "LB_KIM_PRUNED " << lb_kim_prune_rate_current_test / total_1nn_call_current_test << std::endl;
		lb_kim_prune_rate_current_test = 0;
		total_1nn_call_current_test = 0;

		std::cout << "===========================================" << std::endl;
	}

	std::cout << "Total time taken: \t" 
				<< std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - startTime).count() / 1000.0
				<< " s" << std::endl;

		
}

 /// Run classification on test data with all the Ss 
void test_all(std::vector<std::string> datasets, int kfold, const std::vector<double> Ss){

	ofstream file;
	file.open("out.csv", ios::trunc);
	file << "DatasetName";
    for(auto i = Ss.begin(); i != Ss.end() ; i++){
		file << "," << *i;
	}	
	file << std::endl;

	std::chrono::time_point<std::chrono::system_clock> startTime = 
			std::chrono::system_clock::now();
	for(int i = 0 ; i < datasets.size(); i++){
		file << datasets[i] ; 
		std::cout << "======================================" << std::endl;
		std::cout << "Testing " << datasets[i] << " ... " << std::endl;
		std::cout << "======================================" << std::endl;
		std::chrono::time_point<std::chrono::system_clock> thisStartTime = 
			std::chrono::system_clock::now();
		AA::SharedTimeSerieDataset train_dset;
		std::string train_path = datasets[i] + "/" + datasets[i] + "_TRAIN";
		load_dataset(train_path.c_str(), train_dset);
		AA::SharedTimeSerieDataset test_dset;
		std::string test_path = datasets[i] + "/" + datasets[i] + "_TEST";
		load_dataset(test_path.c_str(), test_dset);
		
		std::cout << "Train loading time: \t" 
				<< std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - thisStartTime).count() / 1000.0
				<< " s" << std::endl;
		for( auto sPtr = Ss.begin() ; sPtr != Ss.end() ; sPtr++){
			thisStartTime = std::chrono::system_clock::now();
		/*double best_s = cross_validate(train_dset, kfold, Ss);
		std::cout << "Training time (CV): \t" 
				<< std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - thisStartTime).count() / 1000.0
				<< " s" << std::endl;
		thisStartTime = std::chrono::system_clock::now();
		*/
			// Retransfom train and test with this s
			for(int j = 0 ; j < train_dset.size() ; j++){
				train_dset[j]->setStatesAndJpointsRec(*sPtr, 3);
			}
			double train_w_mean, train_w_sdev;
			AA::normalizeWeightOnDataset(train_dset, &train_w_mean, &train_w_sdev);
			for(int j = 0 ; j < test_dset.size() ; j++){
				test_dset[j]->setStatesAndJpointsRec(*sPtr, 3);
				test_dset[j]->normalizeWeight(train_w_mean, train_w_sdev);
			}
			double acc = validate(train_dset,test_dset);
				
			std::cout <<  "\tS = " << *sPtr << " -> \tTEST ACCURACY = " << acc << "\t ERROR = " << 1 - acc ;
			std::cout << "\t\t TIME: \t" 
				<< std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - thisStartTime).count() / 1000.0
				<< " s " ;
			std::cout  << std::setprecision(6) << "\t LB_KIM_PRUNED " << lb_kim_prune_rate_current_test / total_1nn_call_current_test << std::endl << setprecision(4);
			lb_kim_prune_rate_current_test = 0;
			total_1nn_call_current_test = 0;
			
			file << "," << 1 - acc;
			std::cout << "-------" << std::endl;
		}
		file << std::endl;	
		std::cout << "===========================================" << std::endl;
	}

	std::cout << "Total time taken: \t" 
				<< std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - startTime).count() / 1000.0
				<< " s" << std::endl;

		
	file.close();
}

int main(int argc, char **argv){

// CROSS VALIDATE ONE TEST	

	vector<double> Ss;
	char *token = std::strtok(argv[1],","); 
	double tmpS;
	while(token != NULL){
		sscanf(token, "%lf", &tmpS);
		Ss.push_back(tmpS);
		token = std::strtok(NULL,",");
	}
	std::cout << fixed << setprecision(4) << right;
	
	std::chrono::time_point<std::chrono::system_clock> startTime = 
			std::chrono::system_clock::now();
#ifndef TEST_ALL
 
	AA::SharedTimeSerieDataset dataset;
	AA::SharedTimeSerieDataset test_dataset;
	load_dataset(argv[2], dataset);	
	load_dataset(argv[3], test_dataset);	
	//SharedTimeSerieDataset holdout {dataset.begin(), dataset.begin() + 20};
	//for(auto t : holdout) std::cerr << *t;
	//dataset.erase(dataset.begin(), dataset.begin() + 20);
	//for(auto t : dataset) std::cerr << *t;
	//std::cerr << "acc = " << validate(dataset, holdout) << std::endl;
	//int label = classify_1nn(*holdout[0], dataset);
	//cerr << "label = " << label << std::endl;
	//vector<double> Ss (ss_arr , ss_arr + sizeof(ss_arr) / sizeof(ss_arr[0]));
	//std::string dsets[] = {"Gun_Point"};
	//std::vector<std::string> datasets{dsets , dsets + 1};
	//test_all(datasets, 3,Ss);	
	//std::cerr << "BEST_S = " << best_s << std::endl;	
	std::chrono::time_point<std::chrono::system_clock> startTimeThis = startTime; 
#ifdef CV	
	double s = cross_validate(dataset,3,Ss);
#else 
	for(auto sPtr = Ss.begin() ; sPtr != Ss.end() ; sPtr++){
		double s = *sPtr;	
#endif	
		for(int j = 0 ; j < dataset.size() ; j++){
				dataset[j]->setStatesAndJpointsRec(s, 3.0);
		}
		double w_mean, w_sdev;
		AA::normalizeWeightOnDataset(dataset, &w_mean, &w_sdev);
		//AA::printDataset(dataset , std::cout);
		for(int j = 0 ; j < test_dataset.size() ; j++){
				test_dataset[j]->setStatesAndJpointsRec(s, 3.0);
				test_dataset[j]->normalizeWeight(w_mean, w_sdev);
				//std::cout << *test_dataset[j];
		}
		//std::cerr << " Training W_mean = " << w_mean << " W_sdev = " << w_sdev << std::endl;
		double acc = validate(dataset,test_dataset);
		/*std::cout << "\tACC = " << acc << std::endl;
		std::cout << "Time taken: \t" 
					<< std::chrono::duration_cast<std::chrono::milliseconds>
					(std::chrono::system_clock::now() - startTime).count() / 1000.0
					<< " s" << std::endl;
		*/
		std::cout <<  "\tS = " << s << " -> \tTEST ACCURACY = " << acc << "\t ERROR = " << 1 - acc ;
		std::cout << "\t\t TIME: \t" 
			<< std::chrono::duration_cast<std::chrono::milliseconds>
			(std::chrono::system_clock::now() - startTimeThis).count() / 1000.0
			<< " s" ;
		std::cout  << std::setprecision(6) << "\t LB_KIM_PRUNED " << lb_kim_prune_rate_current_test / total_1nn_call_current_test << std::endl << setprecision(4);
		lb_kim_prune_rate_current_test = 0;
		total_1nn_call_current_test = 0;
		

		std::cout << "-------" << std::endl;
#ifndef CV
		startTimeThis = std::chrono::system_clock::now();
	}
#endif


#else
//TEST ALL

	//vector<double> Ss (ss_arr , ss_arr + sizeof(ss_arr) / sizeof(ss_arr[0]));
	vector<string> datasets_vec (datasets , datasets + sizeof(datasets) / sizeof(datasets[0]));	
#ifdef CV
	cv_test_all(datasets_vec,3,Ss);	
#else
	test_all(datasets_vec,3,Ss);
#endif	
#endif

	std::cout << "TOTAL AVERAGE LB_KIM_PRUNED = \t" << lb_kim_prune_rate_sum / total_1nn_call << std::endl;
	std::cout << "TOTAL TIME = \t" <<  std::chrono::duration_cast<std::chrono::milliseconds>
			(std::chrono::system_clock::now() - startTime).count() / 1000.0
			<< " s"  << std::endl;	
	return 0;
}

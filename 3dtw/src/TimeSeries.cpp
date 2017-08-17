#include "TimeSeries.h"
#include <sstream>
#include <iostream>
#include <string>
#include <cmath>
#include <cassert>
#include <algorithm>

#define POW2(X) ((X) * (X))
#define JDIST(X,Y) (POW2((X).v - (Y).v) + POW2((X).w - (Y).w))
#define SQRT_JDIST(X,Y) (std::sqrt(JDIST((X),(Y))))
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y) )
#define MIN3(X,Y,Z) MIN((MIN(X,Y)),Z)
#define REF_DIST(X,Y) (std::sqrt( (POW2(X) + POW2(Y)) ))
#define INF 1e20

double AA::TimeSerie::operator[](std::size_t index) const{
	return m_data[index];
}

std::size_t AA::TimeSerie::size() const{
	return m_data.size();
}

int AA::TimeSerie::label() const{
	return m_label;
}

void AA::TimeSerie::setLabel(int label) {
	this->m_label = label;
}

//double AA::TimeSerie::w_sum() const { return m_w_sum; }
//double AA::TimeSerie::w_sum2() const { return m_w_sum2; }
int AA::TimeSerie::jpoints_length() const { return static_cast<int>(m_jpoints.size()); }



void AA::TimeSerie::initializeFromCSVRow(std::istream& csvrow, bool norm){
	std::string line;
	
	std::getline(csvrow, line);
	if(line.empty()){
		return;
	}
	std::stringstream lineStream(line);
	std::string cell;
	m_data.clear();
	m_sum = m_sum2 = m_mean = 0;
	bool read_label = false;
	while(std::getline(lineStream, cell, ',')){
		if(!read_label){
			m_label = std::stoi(cell);
			read_label = true;
		}else{
			double val = std::stod(cell);
			m_data.push_back(val);
			m_state.push_back(0); // class invariant |m_state| = |m_data|
			m_sum += val;
			m_sum2 += val * val;
		}
	}
	m_mean = m_sum / this->size();
	//std::cerr << "m_sum = " << m_sum << ", mean = " << mean << std::endl;
	m_sdev = std::sqrt(m_sum2/this->size() - m_mean * m_mean );
	this->m_normed = false;
	if(norm){
		// calculate z-score
		this->normalize();
	}
}

void AA::TimeSerie::normalize(){
	if(!this->m_normed){
		double inverseSDev = 1.0 / (this->m_sdev > 0 ? this->m_sdev : 1.0);
		if(this->m_mean){
			for(std::size_t i = 0 ; i < this->size() ; i++){
				this->m_data[i] = (this->m_data[i] - this->m_mean) * inverseSDev;
			}
			this->m_mean = 0.0;
		}else if(inverseSDev != 1.0){
			for(std::size_t i = 0 ; i < this->size() ; i++){
				this->m_data[i] *= inverseSDev;
			}
		}
		this->m_sdev = 1.0;
		this->m_normed = true;
	}
}
void AA::TimeSerie::setStatesAndJpoints(double ss, double K){

	//std::cerr << "Setting states .. mean = " << this->m_mean << ", m_sdev = " << 
	//									this->m_sdev << std::endl;
	m_jpoints.clear();
	m_w_sum = 0;
	m_w_sum2 = 0;
	if(m_data.empty()) return;
	Jpoint currJmp;
	currJmp.s = 0; // <<-- t_0 as J_0 
	double currSectionSum = 0;
	for(std::size_t i = 0; i < this->size(); i++ ){
		if(m_data[i] > K)
			m_state[i] = K + ss;
		else if(m_data[i] <= -1 * K)
			m_state[i] = -1 * K;
		else{
			for(double j = -1 * K; j < K ; j += ss){
				if(m_data[i] > j && m_data[i] <= j + ss ){
					m_state[i] = j + ss;
					break;
				}
			}	
		}
		// test if jpoint 
		if( i > 0){
			if( m_state[i] != m_state[i-1] || i == this->size() - 1 ){	
				// close current segment and make new jump point
				currJmp.w = i - currJmp.s;
				m_w_sum += currJmp.w; 
				m_w_sum2 += currJmp.w * currJmp.w;	
				currJmp.v = currSectionSum / currJmp.w;
				currSectionSum = 0;	
				m_jpoints.push_back(currJmp);
				currJmp.s = i;	// <<-- new start for new jump point
			}
		}else{
			assert( i == 0);
			if(i == this->size() - 1) // m_data has only one element t_0
			{
				currJmp.v = m_data[0];
				currJmp.w = 0;
				m_jpoints.push_back(currJmp);
			}
		}

		currSectionSum += m_data[i]; 
	}
	
	//std::cout << "Size of Jpoints set = " << this->m_jpoints.size() << std::endl;
	
	//weights changed so they need to be renormalized
	this->m_weights_normed = false;	
}

void AA::TimeSerie::setStatesAndJpointsRec(double ss, double K){

#ifdef S_RECURSIVE	
	m_jpoints.clear();
	m_w_sum = 0;
	m_w_sum2 = 0;
	setStatesAndJpointsRec(ss, K, 0, static_cast<int>(m_data.size()));
//	std::cout << "Size of Jpoints set = " << this->m_jpoints.size() << std::endl;
#else
	setStatesAndJpoints(ss,K);
#endif
}

/// recursively set states and jpoints between l and s index int datapoints
// l: first index
// s: last index
// jl: first jpoint index to replace
// js: last jpoint index to replace
void AA::TimeSerie::setStatesAndJpointsRec(double ss, double K, int l, int s){

	//std::cerr << "Setting states .. mean = " << this->m_mean << ", m_sdev = " << 
	//									this->m_sdev << std::endl;
	//std::cerr << " ### setting states with S = " << ss <<
	//						" on datas [" << l << "," << s <<")" << std::endl;
	if(l == s) return;
	//auto insertPos = m_jpoints.end() - 1;
	/*for(int i = js; i < jl; i++){
		auto erased = m_jpoints.erase(m_jpoints.begin()+i);
		m_w_sum -= erased->w;
		m_w_sum2 -= POW2(erased->w);
	}*/
	Jpoint currJmp;
	currJmp.s = l; // <<-- t_l as J_l 
	double currSectionSum = 0;
	double currSectionSum2 = 0;
	double currSectionSDev = 0;

	for(std::size_t i = l; i < this->size() && static_cast<int>(i) < s; i++ ){
		if(m_data[i] > K)
			m_state[i] = K + ss;
		else if(m_data[i] <= -1 * K)
			m_state[i] = -1 * K;
		else{
			for(double j = -1 * K; j < K ; j += ss){
				if(m_data[i] > j && m_data[i] <= j + ss ){
					m_state[i] = j + ss;
					break;
				}
			}	
		}
		// test if jpoint 
		if( static_cast<int>(i) > l){
			if( m_state[i] != m_state[i-1] || static_cast<int>(i) == s - 1 ){	
				// close current segment and make new jump point
				currJmp.w = i - currJmp.s;
				if(currJmp.w == 0){
					currJmp.v =currSectionSum;
					currSectionSDev = 0;
				}else{
					currJmp.v = currSectionSum / currJmp.w;
					currSectionSDev = std::sqrt( 
								(currSectionSum2 / currJmp.w) - POW2(currJmp.v));
				}
				currSectionSum = 0;	
				currSectionSum2 = 0;	
		
				if(currSectionSDev > .5){
					setStatesAndJpointsRec(.1 , K, currJmp.s, i);
				}else{	// jump point is fixed!
					m_w_sum += currJmp.w; 
					m_w_sum2 += currJmp.w * currJmp.w;	
					m_jpoints.push_back(currJmp); // insert at insert point and move the insertPos one position forward.
					//fprintf(stderr, "New jpoint (s=%lf) [%d,%d), v = %lf, w = %lf\n",ss,currJmp.s,i,currJmp.v, currJmp.w);
				}
				currJmp.s = i;	// <<-- new start for new jump point
			}
		}else{
			assert( i == l);
			if(static_cast<int>(i) == s-1) // m_data has only one element t_0
			{
				currJmp.v = m_data[i];
				currJmp.w = 0;
				m_jpoints.push_back(currJmp); // insert at insert point and move the insertPos one position forward.
			}
		}
		
		// It's not a jump point
		currSectionSum += m_data[i]; 
		currSectionSum2 += POW2(m_data[i]); 
	}
	
	//std::cout << "Size of Jpoints set = " << this->m_jpoints.size() << std::endl;
	
	//weights changed so they need to be renormalized
	this->m_weights_normed = false;	
}

void AA::TimeSerie::normalizeWeight(double w_mean, double w_sdev){
	if(this->m_weights_normed)	
		return;
	for(std::size_t i = 0 ; i < m_jpoints.size() ; i++){
		//std::cerr << "m_jpoint.w before " << m_jpoints[i].w << " ->";
		m_jpoints[i].w = (m_jpoints[i].w - w_mean ) / w_sdev; 	
		//std::cerr << "m_jpoint.w after " << m_jpoints[i].w << std::endl;
	}
	this->m_weights_normed = true;
}

double AA::distance(const AA::TimeSerie& p, const AA::TimeSerie& q, double bfs){
	if(p.m_jpoints.empty() || q.m_jpoints.empty())
		return std::numeric_limits<double>::infinity();
	
	//std::cerr << "Making DTW matrix : " << p.m_jpoints.size() << " X " << q.m_jpoints.size() << std::endl;
	std::vector<std::vector<double> > D(p.m_jpoints.size(), std::vector<double>(q.m_jpoints.size(),0));
	//for(std::size_t i = 0 ; i < D.size(); i++)
	//	D[i].resize(q.m_jpoints.size()); 
	D[0][0] = JDIST(p.m_jpoints[0],q.m_jpoints[0]);
	for(std::size_t i = 1 ; i < q.m_jpoints.size() ; i++){ // first row
		D[0][i] = JDIST(p.m_jpoints[0],q.m_jpoints[i]) + D[0][i-1];
	}
	for(std::size_t i = 1 ; i < D.size() ; i++){ // first column
		D[i][0] = JDIST(p.m_jpoints[i],q.m_jpoints[0]) + D[i-1][0];
	}

	for(std::size_t i = 1 ; i < D.size() ; i++){
		double min_cost = INF;
		for(std::size_t j = 1 ; j < q.m_jpoints.size() ; j++){
			D[i][j] = JDIST(p.m_jpoints[i], q.m_jpoints[j]) + 
					MIN3(D[i][j-1] , D[i-1][j] , D[i-1][j-1]) ;
			if( D[i][j] < min_cost)
				min_cost = D[i][j];
		}
		if(min_cost >= bfs){
			//std::cerr << "EARLY ABONDAN DTW" << std::endl;
			return min_cost;
		}
	}

	return D[p.m_jpoints.size()-1][q.m_jpoints.size()-1];
}

std::istream& AA::operator>>(std::istream& str, TimeSerie& ts){
	ts.initializeFromCSVRow(str,false);
	return str;
}

std::ostream& AA::operator<<(std::ostream& str, const TimeSerie& ts){
	str << "T = " << ts.m_label << " : [ ";
	for(std::size_t i = 0; i < ts.size(); i++){
		str << "(" << ts.m_data[i] << " : " << ts.m_state[i] <<"), ";
	}
	str << "]" << std::endl;
	str << "T.Jpoints (start,duration, value)= [ "; 
	for(std::size_t i = 0; i < ts.m_jpoints.size(); i++){
		str << "("	<< ts.m_jpoints[i].s << " , " 
					<< ts.m_jpoints[i].w << " , "
					<< ts.m_jpoints[i].v << " ), ";
	}
	str << "]" << std::endl;
	return str;
}

void AA::normalizeWeightOnDataset(const SharedTimeSerieDataset& dataset, double *w_mean, double *w_sdev){
	double total_w_sum = 0;
	double total_w_sum2 = 0;
	int total_jpoints = 0;
	for(std::size_t i = 0 ; i < dataset.size(); i++){
		total_w_sum += dataset[i]->m_w_sum;
		total_w_sum2 += dataset[i]->m_w_sum2;
		total_jpoints += dataset[i]->jpoints_length();
	}
	double dataset_w_mean = total_w_sum / total_jpoints;
	double dataset_w_sdev = std::sqrt( total_w_sum2 / total_jpoints - 
						dataset_w_mean * dataset_w_mean);
	for(std::size_t i = 0 ; i < dataset.size(); i++){
		dataset[i]->normalizeWeight(dataset_w_mean, dataset_w_sdev); 
	}
	if(w_mean != NULL)
		*w_mean = dataset_w_mean;
	if(w_sdev != NULL)
		*w_sdev = dataset_w_sdev;
}

void AA::printDataset(const SharedTimeSerieDataset& dataset, std::ostream& str){
	for(int j = 0 ; j < dataset.size() ; j++){
			str << *dataset[j];
		}
}

double AA::lb_kim_hierarchy(const AA::TimeSerie& t, const AA::TimeSerie& q, double bsf)
{
    /// 1 point at front and back
    double d, lb;
	int t_len = t.m_jpoints.size();
	int q_len = q.m_jpoints.size();
	auto tx0 = t.m_jpoints.cbegin();
	auto ty0 = t.m_jpoints.cend() - 1;

	auto qx0 = q.m_jpoints.cbegin();
	auto qy0 = q.m_jpoints.cend() - 1;

	lb = JDIST(*tx0,*qx0);
	//std::cerr << "stop 0" << std::endl;	
	if(t_len < 2 || q_len < 2) 	return lb;

	//std::cerr << "stop 1" << std::endl;	

    lb += JDIST(*ty0,*qy0);
    if (lb >= bsf)   return lb;

	//std::cerr << "stop 2" << std::endl;	
	if(t_len < 3 || q_len < 3) return lb;
	//std::cerr << "stop 3" << std::endl;	
    /// 2 points at front
    auto tx1 = tx0 + 1;
	auto qx1 = qx0 + 1;
    d = MIN3(JDIST(*tx1,*qx0), JDIST(*tx0,*qx1), JDIST(*tx1,*qx1));
    lb += d;
    if (lb >= bsf)   return lb;

	//std::cerr << "stop 4" << std::endl;	
	if(t_len < 4 || q_len < 4) return lb;

	//std::cerr << "stop 5" << std::endl;	
    /// 2 points at back
	auto ty1 = ty0 - 1;
	auto qy1 = qy0 - 1; 
    d = MIN3(JDIST(*ty1,*qy0), JDIST(*ty0, *qy1), JDIST(*ty1,*qy1));
    lb += d;
    if (lb >= bsf)   return lb;

	if(t_len < 5 || q_len < 5) return lb;
    /// 3 points at front
	auto tx2 = tx1 + 1;
	auto qx2 = qx1 + 1;
    d = MIN3(JDIST(*tx0,*qx2), JDIST(*tx1, *qx2), JDIST(*tx2,*qx2));
    d = MIN3(d, JDIST(*tx2,*qx1), JDIST(*tx2,*qx0));
    lb += d;
    if (lb >= bsf)   return lb;

	if(t_len < 6 || q_len < 6) return lb;
    /// 3 points at back
	auto ty2 = ty1 - 1;
	auto qy2 = qy1 - 1; 
    d = MIN3(JDIST(*ty0,*qy2), JDIST(*ty1, *qy2), JDIST(*ty2,*qy2));
    d = MIN3(d, JDIST(*ty2,*qy1), JDIST(*ty2,*qy0));
    lb += d;

    return lb;
}

double AA::lb_kim_minmax(const AA::TimeSerie& t, const AA::TimeSerie& q, double bsf)
{
	// IN PROGRESS
    /// 1 point at front and back
    double d, lb;
	int t_len = t.m_jpoints.size();
	int q_len = q.m_jpoints.size();
	auto tx0 = t.m_jpoints.cbegin();
	auto ty0 = t.m_jpoints.cend() - 1;

	auto qx0 = q.m_jpoints.cbegin();
	auto qy0 = q.m_jpoints.cend() - 1;

	lb = JDIST(*tx0,*qx0);
	//std::cerr << "stop 0" << std::endl;	
	if(t_len < 2 || q_len < 2) 	return lb;

	//std::cerr << "stop 1" << std::endl;	

    lb += JDIST(*ty0,*qy0);
    if (lb >= bsf)   return lb;

	if(t_len < 3 || q_len < 3) return lb;
/*	
	//std::cerr << "stop 3" << std::endl;	
    /// 2 points at front
    auto tx1 = tx0 + 1;
	auto qx1 = qx0 + 1;
    d = MIN3(JDIST(*tx1,*qx0), JDIST(*tx0,*qx1), JDIST(*tx1,*qx1));
    lb += d;
    if (lb >= bsf)   return lb;

	//std::cerr << "stop 4" << std::endl;	
	if(t_len < 4 || q_len < 4) return lb;

	//std::cerr << "stop 5" << std::endl;	
    /// 2 points at back
	auto ty1 = ty0 - 1;
	auto qy1 = qy0 - 1; 
    d = MIN3(JDIST(*ty1,*qy0), JDIST(*ty0, *qy1), JDIST(*ty1,*qy1));
    lb += d;
    if (lb >= bsf)   return lb;

	if(t_len < 5 || q_len < 5) return lb;
    /// 3 points at front
	auto tx2 = tx1 + 1;
	auto qx2 = qx1 + 1;
    d = MIN3(JDIST(*tx0,*qx2), JDIST(*tx1, *qx2), JDIST(*tx2,*qx2));
    d = MIN3(d, JDIST(*tx2,*qx1), JDIST(*tx2,*qx0));
    lb += d;
    if (lb >= bsf)   return lb;

	if(t_len < 6 || q_len < 6) return lb;
    /// 3 points at back
	auto ty2 = ty1 - 1;
	auto qy2 = qy1 - 1; 
    d = MIN3(JDIST(*ty0,*qy2), JDIST(*ty1, *qy2), JDIST(*ty2,*qy2));
    d = MIN3(d, JDIST(*ty2,*qy1), JDIST(*ty2,*qy0));
    lb += d;

    if (lb >= bsf)   return lb;

	if(t_len < 7 || q_len < 7) return lb;
*/
	double min_t_d = INF;
	double max_t_d = 0;
	for(int i = 3 ; i < t_len -3 ; i++){
		auto j = t.m_jpoints[i];
		double d = REF_DIST(j.v,j.w);
		if(d < min_t_d)
			min_t_d = d;
		if(d > max_t_d)
			max_t_d = d;	
	}
	
	double min_q_d = INF;
	double max_q_d = 0;
	for(int i = 3 ; i < q_len -3 ; i++){
		auto j = q.m_jpoints[i];
		double d = REF_DIST(j.v,j.w);
		if(d < min_q_d)
			min_q_d = d;
		if(d > max_q_d)
			max_q_d = d;	
	}

	lb+= POW2(min_t_d - min_q_d) ;
	if(max_t_d != min_t_d && max_q_d != min_q_d)
		lb+= POW2(max_t_d - max_q_d);

	return lb;

}



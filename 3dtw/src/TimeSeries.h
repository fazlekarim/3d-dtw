#include <vector>
#include <istream>
#include <memory>

#define INF 1e20

namespace AA{

class TimeSerie;
// Different datasets of timeseries share the same objects to enable cheap shallow copying
typedef std::vector<std::shared_ptr<AA::TimeSerie>> SharedTimeSerieDataset;

class TimeSerie{
	public:
		double operator[](std::size_t index) const;
		std::size_t size() const;
		void initializeFromCSVRow(std::istream& row, bool normed = true);
		void normalize();
		void setStatesAndJpoints(double ss, double K = 3);
		void setStatesAndJpointsRec(double ss, double K);
		void setStatesAndJpointsRec(double ss, double K ,int,int);
		void normalizeWeight(double w_mean, double w_sdev);
		int label() const;
		//const std::vector<Jpoint>& jpoints() const;
		void setLabel(int label);
		int jpoints_length() const;
		//double w_sum() const;	//return sum of weights of jpoints
		//double w_sum2() const; 	//return sum of squares of weights in jpoints
	private:
		std::vector<double> m_data;
		std::vector<double> m_state;
		int m_label;
		double m_sum = 0;
		double m_sum2 = 0; // sum pf squares
		double m_mean = 0;
		double m_sdev = 0;

		typedef struct jpoint{
			int s;	// start point
			double w;	// duration/weight
			double v;	// value
		} Jpoint;
		std::vector<Jpoint> m_jpoints;
		double m_w_sum = 0;
		double m_w_sum2 = 0;
			
		bool m_normed = false;
		bool m_weights_normed = false;	// true iff weights are normalized 
										// and states were not reset since after
		double inline jpoint_dist(const Jpoint& j1, const Jpoint& j2);
		friend std::ostream& operator<<(std::ostream&, const TimeSerie& ts);
		friend double distance(const TimeSerie&, const TimeSerie&, double max_thresh);
		friend void normalizeWeightOnDataset(const AA::SharedTimeSerieDataset&, 
				double *w_mean, double*w_sdev);

		friend double lb_kim_hierarchy(const AA::TimeSerie&, const AA::TimeSerie&, double bsf);
		friend double lb_kim_minmax(const AA::TimeSerie&, const AA::TimeSerie&, double bsf);
};

std::istream& operator>>(std::istream& str, TimeSerie& ts);	
std::ostream& operator<<(std::ostream& str, const TimeSerie& ts);	


double distance(const TimeSerie& p, const TimeSerie& q, double max_thresh);

double lb_kim_hierarchy(const AA::TimeSerie&, const AA::TimeSerie&, double bsf = INF);
double lb_kim_minmax(const AA::TimeSerie&, const AA::TimeSerie&, double bsf = INF);

void normalizeWeightOnDataset(const SharedTimeSerieDataset& dataset, double *w_mean , double *w_sdev);
void printDataset(const SharedTimeSerieDataset& dataset, std::ostream& );

}

//This code was used to perform the simulations in the paper `Degree irregularity and rank probability bias in network meta-analysis' A L Davies, T Galla (2020) 
//The code simulates binomial data for a network meta-analysis with a combination of two-arm and multi-arm trials
//A Bayesian NMA is performed for each realisation of the data and the results are averaged
//The Bayesian NMA uses the Metropolis-in-Gibbs algorithm described in Spiegelhalter, Thomas, Best and Lunn (2003) `WINBGUGS User Manual' 
//The user must define the geometry of the network and the true model parameter values at the beginning of int main () {  }
//This code is written for a network of 4 treatments which are equally effective - changes to the code are required for a network of more/fewer treatments and/or non-equally effective treatments - these changes are commented where possible (NB: For different number of treatments, the user must change the number of model parameter vectors which store values at each iteration - we do not comment on this throughout the code)
//The code relies on the Eigen library, the download files and documentation can be found here: http://eigen.tuxfamily.org/index.php?title=Main_Page
//The generation of random numbers (outside of functions) uses the PCG library, the download files and documentation can be found here: https://www.pcg-random.org/
//Code written by Annabel Davies 09/2019
//annabel.davies@postgrad.manchester.ac.uk

#include <iostream>
#include<Eigen/Dense>
#include<Eigen/LU>
#include<random>
#include<cmath>
#include<fstream>
#include<sstream>
#include<string> 
#include <algorithm>
#include <cstdlib>
#include "pcg_random.hpp"

using namespace std;
using namespace Eigen;

//Simulation functions*****************************************************************************
//a class structure to generate multivariate normal random variables
struct normal_random_variable
{
	normal_random_variable(Eigen::MatrixXd const& covar)
		: normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar)
	{}

	normal_random_variable(Eigen::VectorXd const& mean, Eigen::MatrixXd const& covar)
		: mean(mean)
	{
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
		transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
	}

	Eigen::VectorXd mean;
	Eigen::MatrixXd transform;

	Eigen::VectorXd operator()() const
	{
		static std::mt19937 gen{ std::random_device{}() };
		static std::normal_distribution<> dist;

		return mean + transform * Eigen::VectorXd{ mean.size() }.unaryExpr([&](double x) { return dist(gen); });
	}
};

//*************************************************************************************************

//NMA Functions************************************************************************************
//Using Bayes' theorem we can construct the posterior distributions of each model parameter conditional on all other parameters. We then cycle through these conditional distributions using the Metropolis-in-Gibbs method. At each Metropolis-Hastings step we decide whether to accept the proposed new parameter value based on the logs of the probability density functions at the current and proposed values. The following four functions calculate these log(p) values for the different model parameters, delta (trial specific treatment effects), tau (heterogeneity parameter), d (treatment effects) and b (trial specific baseline effects). 

//function to find ln(p(delta))
//log of the probability density of the trial-specific treatment effects
double ln_delta_density(int i, double delta_sc, double b, int r_alpha, int n_alpha, double tau, VectorXd delta_vec, VectorXd d, VectorXi a) {
	//nb: i is trial number - 1 (or trial number if you count from 0)
	//r_alpha and n_alpha are the number of events and number of participants in arm-alpha of trial i 
	//delta_sc is the trial specific treatment effect of treatment alpha relative to the trial specific baseline in trial i 
	//b is the baseline treatment effect in trial i 
	//delta_vec is the vector of trial specific treatment effects  
	//d is the vector of mean treatment effects
	//a is the vector cotaining the number of arms per trial

	//cast integers to doubles
	double r = (double)r_alpha;
	double n = (double)n_alpha;

	//covariate matrix = Omega_i = tau^2 * C_i 
	//create C_i 
	MatrixXd C_i(a(i) - 1, a(i) - 1); //dimensions where a(i) = number of arms in study i
	
	//diagonal elements of C_i = 1
	C_i = MatrixXd::Identity(a(i) - 1, a(i) - 1);
	//off diagonal elemnts of C_i = 1/2
	for (int r = 0;r < C_i.rows();r++) {
		for (int c = 0;c < C_i.cols();c++) {
			if (C_i(r, c) != 1) {
				C_i(r, c) = 0.5;
			}
			else {
				C_i(r, c) = C_i(r, c);
			}
		}
	}
	

	//Covariate matrix = Omega_i = tau^2*C_i
	MatrixXd Omega = pow(tau, 2.0)*C_i;

	//initialise trial-level vectors of delta and d 
	//delta_i and d_i - size (a_i-1)
	VectorXd delta_i(a(i) - 1);
	VectorXd d_i(a(i) - 1);

	//find sum of a(i) up to i 
	int sum_a = 0;
	for (int j = 0; j <= i; j++) {
		sum_a = sum_a + a(j);
	}
	
	//give values to delta_i and d_i using delta and d
	for (int k = 1; k < a(i);k++) {
		delta_i(a(i) - 1 - k) = delta_vec(sum_a - k);
		d_i(a(i) - 1 - k) = d(sum_a - k);
	}

	//delta_i-d_i
	VectorXd diff = delta_i - d_i;
	
	//work out each term of ln(p(delta)) 
	//binomial:
	double term_1 = r * (b + delta_sc);
	double term_2 = -n * log(1.0 + exp(b + delta_sc));
	//multivariate normal:
	double term_3 = -0.5*diff.transpose()*Omega.inverse()*diff;

	double ln_density = term_1 + term_2 + term_3;
	return ln_density;
}

//function to get ln(p(tau))
//log of the probability density of the heterogeneity parameter
double ln_tau_density(double tau, VectorXd delta, VectorXd d, VectorXi a, int n_s) {
	//n_s is the number of studies in the network
	//vector delta has zero as first element of each section 
	double ln_density = 0.0;
	for (int i = 0;i < n_s;i++) {

		//create C_i (where Omega_i=tau^2*C_i)
		MatrixXd C_i(a(i) - 1, a(i) - 1);//dimensions where a(i) = no. of arms in trial i

		//diagonal elements of C_i = 1
		C_i = MatrixXd::Identity(a(i) - 1, a(i) - 1);
		//off diagonal elements of C_i = 1/2
		for (int r = 0;r < C_i.rows();r++) {
			for (int c = 0;c < C_i.cols();c++) {
				if (C_i(r, c) != 1) {
					C_i(r, c) = 0.5;
				}
				else {
					C_i(r, c) = C_i(r, c);
				}
			}
		}

		//initialise delta_i and d_i - size (a_i-1)
		VectorXd delta_i(a(i) - 1);
		VectorXd d_i(a(i) - 1);

		//find sum of a(i) up to i 
		int sum_a = 0;
		for (int j = 0; j <= i; j++) {
			sum_a = sum_a + a(j);
		}
		
		//give values to delta_i and d_i using delta and d
		for (int k = 1; k < a(i);k++) {
			delta_i(a(i) - 1 - k) = delta(sum_a - k);
			d_i(a(i) - 1 - k) = d(sum_a - k);
		}
		
		//Omega_i = tau^2 * C_i
		MatrixXd Omega = pow(tau, 2.0)*C_i;
		
		//first term of log(p(tau))
		double term_1 = -0.5*log(Omega.determinant());

		//second term of log(p(tau))
		VectorXd diff = delta_i - d_i;
		double term_2 = -0.5*diff.transpose()*Omega.inverse()*diff;
		ln_density = ln_density + term_1 + term_2;
	}

	return ln_density;
}

//function to get ln(p(d))
//log of the probability density of the treatment effects
double ln_d_density(double d_1a, double tau, VectorXd d, VectorXd delta, int n_s, VectorXi a, double s_d) {
	//s_d is the standard deviation of the prior distribution of the treatment effect
 
	//prior for d
	double ln_density = -pow(d_1a, 2.0) / (2.0 * pow(s_d, 2.0));
	
	for (int i = 0;i < n_s;i++) {
	
		//create C_i where Omega_i = tau^2 * C_i 
		MatrixXd C_i(a(i) - 1, a(i) - 1); //dimensions where a(i) = no. of arms in trial i
		//diagonal elements of C_i = 1
		C_i = MatrixXd::Identity(a(i) - 1, a(i) - 1);
		//off diagonal elements of C_i = 1/2
		for (int r = 0;r < C_i.rows();r++) {
			for (int c = 0;c < C_i.cols();c++) {
				if (C_i(r, c) != 1) {
					C_i(r, c) = 0.5;
				}
				else {
					C_i(r, c) = C_i(r, c);
				}
			}
		}

		//initialise delta_i and d_i - size (a_i-1)
		VectorXd delta_i(a(i) - 1);
		VectorXd d_i(a(i) - 1);

		//find sum of a(i) up to i 
		int sum_a = 0;
		for (int j = 0; j <= i; j++) {
			sum_a = sum_a + a(j);
		}
		
		//give values to delta_i and d_i sing delta and d
		for (int k = 1; k < a(i);k++) {
			delta_i(a(i) - 1 - k) = delta(sum_a - k);
			d_i(a(i) - 1 - k) = d(sum_a - k);
		}
		
		//Omega_i = tau^2 * C_i
		MatrixXd Omega = pow(tau, 2.0)*C_i;
		
		VectorXd diff = delta_i - d_i;
		ln_density = ln_density - 0.5*diff.transpose()*Omega.inverse()*diff;
	}
	return ln_density;
}

//function to get ln(p(b))
//log of the probability density of the baseline treatment effects
double ln_b_density(int trial, double b_i, double s_b, VectorXi r, VectorXi n, VectorXi a, VectorXd delta) {
	//s_b is the standard deviation of the prior distribution of the baseline effects
	//r and n are the vectors of the number of events and number of participants of each trial

	//Trial is specified from 1 to n_s
	//Vector delta has a zero as first element (so it has same dimensions as r and n)
	
	//find r_ti1 and n_ti1
	//find element ti1 = a_1+a_2+... up to i
	int t_1 = 0;
	for (int i = 0; i < trial - 1; i++) {
		t_1 = t_1 + a(i);
	}
	
	//cast ints to doubles 
	double r_1 = (double)r(t_1);
	double n_1 = (double)n(t_1);

	//calculate the terms of ln(p(b))
	double term_1 = r_1 * b_i;
	double term_2 = -n_1 * log(1.0 + exp(b_i));
	double term_3 = -pow(b_i, 2.0) / (2.0 * pow(s_b, 2.0));
	double term_4 = 0.0;
	//now sum from element t_2 to a_i
	for (int j = t_1 + 1; j < t_1 + a(trial - 1);j++) {
		//cast int to double 
		double r_j = (double)r(j);
		double n_j = (double)n(j);

		term_4 = term_4 + r_j * (b_i + delta(j)) - n_j * log(1.0 + exp(b_i + delta(j)));

	}

	double ln_density = term_1 + term_2 + term_3 + term_4;
	return ln_density;
}

//function to perform Metropolis Hastings step for b, delta and d
//finds the next value in iteration by comparing ln(p) at the current and proposed value
double MH_nextvalue(double ln_density_current, double ln_density_prop, double value_current, double value_prop) {
	
	//draw a random number u from unif(0,1)
	random_device generator;
	mt19937 twist(generator());
	uniform_real_distribution<double> uni_distribution(0.0, 1.0);
	double u = uni_distribution(twist);

	//if u < alpha = p(prop)/p(current) set value(t+1) = value(proposed) 
	//NB this is the same as the condition ln(u) < ln(p(prop))-ln(p(current))
	if (log(u) < ln_density_prop - ln_density_current) {
		return value_prop;
	}
	//else stays the same  value(t+1)=value(t)
	else {
		return value_current;
	}

}

//function to perform the Metropolis-Hastings step for the heterogeneity parameter
//this is different from the other parameters as we have to enforce the uniform prior U(0,5)
double MH_nextvalue_tau(double tau, double tau_prop, VectorXd d, int n_s, VectorXd delta, VectorXi a) {
	
	//Check that the proposed value is within the limit prescribed by the uniform prior distribution U(0,5)
	if (tau_prop < 0.0 || tau_prop > 5.0) {
		return tau;
	}
	else {
		//draw a random number u from unif(0,1)
		random_device generator;
		mt19937 twist(generator());
		uniform_real_distribution<double> uni_distribution(0.0, 1.0);
		double u = uni_distribution(twist);

		//if u < log(p(tau_prop))-log(p(tau))
		double current = ln_tau_density(tau, delta, d, a, n_s);
		double prop = ln_tau_density(tau_prop, delta, d, a, n_s);
		
		if (log(u) < prop - current) {
			return tau_prop;
		}
		//else stays the same  value(t+1)=value(t)
		else {
			return tau;
		}
	}
}

//*************************************************************************************************

//Beginning of main code***************************************************************************
int main()
{

	//DEFINE Network Parameters****************************************************************
	//This section must be edited to create the specific network 
	int n_t = 4;  			//no. of treatments
	double nt_d = (double)n_t;	//cast to a double

	//'real' values of model parameters to be used in simulating data
	double tau_real = 0.1; 		//'real' value of heterogeneity
	VectorXd d_basic_real(n_t - 1);
	d_basic_real << 0.0, 0.0, 0.0;	//'real' values of basic parameters
	
	//N is the vector describing the number of treatments per comparison
	//for n_t = 4: N = (N_12, N_13, N_14, N_23, N_24, N_34) 
	//N_ab = the number of studies that compare a and b
	int l = n_t * (n_t - 1) / 2;	//length of vector N (no. of treatment combinations)
	double l_d = (double)l;
	//CREATE N
	VectorXi N(l);
	N(0) = 1;
	N(1) = 2;
	N(2) = 3;
	N(3) = 5;
	N(4) = 10;
	N(5) = 15;
		
	//Number of 2,3 and 4 arm trials (edit depending on n_t)
	int n_2 = 30;
	int n_3 = 0;
	int n_4 = 1;
		
	//number of studies is sum over n_i
	int n_s = n_2 + n_3 + n_4;
	
	//vector a defines the number of treatments per study
	VectorXi a(n_s);
	for (int k2 = 0;k2 < n_2;k2++) {
		a(k2) = 2;
	}
	for (int k3 = n_2;k3 < n_2 + n_3;k3++) {
		a(k3) = 3;
	}
	for (int k4 = n_2 + n_3; k4 < n_s;k4++) {
		a(k4) = 4;
	}
	//calculate the sum over a (=the total number of arms/treatment groups in the network)
	int a_sum = 0;
	for (int j = 0;j < n_s;j++) {
		a_sum = a_sum + a(j);
	}


	//create vector of treatments (edit depending on n_t and geometry)
	VectorXi treat_net(a_sum);
	VectorXi N2(l); 	//vector of 2-arm trials only
	N2(0) = 0;
	N2(1) = 1;
	N2(2) = 2;
	N2(3) = 4;
	N2(4) = 9;
	N2(5) = 14;
	//now define the list of treatments in order of trials  
	int sumN = 0;
	for (int j = 0;j < l;j++) {
		for (int i = 0;i < N2(j);i++) {
			if (j < 3) {
				treat_net(2 * sumN + 2 * i) = 1;
				treat_net(2 * sumN + 2 * i + 1) = j+2;
			}
			else if (j >= 3 && j < 5) {
				treat_net(2 * sumN + 2 * i) = 2;
				treat_net(2 * sumN + 2 * i + 1) = j;
			}
			else {
				treat_net(2 * sumN + 2 * i) = 3;
				treat_net(2 * sumN + 2 * i + 1) = 4;
			}
		}
		sumN = sumN + N2(j);
	}
	//now 3-arm trials (as appropriate)
	//Add from final element backwards e.g.
	/*
	//study 234
	treat_net(a_sum-1) = 4;
	treat_net(a_sum-2) = 3;
	treat_net(a_sum-3) = 2;
	//study 134
	treat_net(a_sum-4) = 4;
	treat_net(a_sum-5) = 3;
	treat_net(a_sum-6) = 1;
	//study 124
	treat_net(a_sum-7) = 4;
	treat_net(a_sum-8) = 2;
	treat_net(a_sum-9) = 1;
	//study 123
	treat_net(a_sum-10) = 3;
	treat_net(a_sum-11) = 2;
	treat_net(a_sum-12) = 1;
	*/

	//Now 4-arm trials (as appropriate) 
	//Add from final element backwards
	treat_net(a_sum-1) = 4;
	treat_net(a_sum-2) = 3;
	treat_net(a_sum-3) = 2;
	treat_net(a_sum-4) = 1;

	//Define a vector to store the number of participants per arm
	VectorXi n(a_sum);  
	for (int i = 0; i < a_sum; i++) {
		n(i) = 25;
	}

	//*****************************************************************************************


	//DEFINE Simulation Parameters*************************************************************

	int it = 1000;			//No. of simulation iterations
	int NG = 25000; 		//No. of Gibbs iterations
	int burn = 5000; 		//No. of burn in 
	int thin = 10;			//Thinning factor

	//create doubles of these values
	double dit = (double)it;
	double N_d = (double)NG;
	double burn_d = (double)burn;
	double thin_d = (double)thin;
	double N_new = (N_d - burn_d)/thin_d;	//No. of iterations after burn in and thinning
	
	//Standard deviations of prior distributions of b and d
	double s_b = 10000.0;
	double s_d = 10000.0;

	//At each Gibbs iteration a proposed value is picked from from a normal distribution with mean equal to the current parameter value and SD defined here
	//these values are picked based on Brooks-Gelman-Rubin convergence testing 
	double sd_prop_delta = 0.3;
	double sd_prop_b = 0.5;
	double sd_prop_tau = 0.03;
	double sd_prop_d = 0.07;

	//*****************************************************************************************

	//Files and storage************************************************************************
	
	//create vectors for mean and sd values of each model parameter (d_ab and tau)
	//(d_12, d_13, d_14, d_23, d_24, d_34) = (d0, d1, d2, d3, d4, d5)
	VectorXd vec_mean_d0(it);
	VectorXd vec_sd_d0(it);
	VectorXd vec_mean_d1(it);
	VectorXd vec_sd_d1(it);
	VectorXd vec_mean_d2(it);
	VectorXd vec_sd_d2(it);
	VectorXd vec_mean_d3(it);
	VectorXd vec_sd_d3(it);
	VectorXd vec_mean_d4(it);
	VectorXd vec_sd_d4(it);
	VectorXd vec_mean_d5(it);
	VectorXd vec_sd_d5(it);
	VectorXd vec_mean_tau(it);
	VectorXd vec_sd_tau(it);

	//create vectors of the bias on the probabilities that each treatment has each rank 
	//store this value at each simulation iteration
	//P_Ta_rb = the probability that treatment a has rank b
	VectorXd P_T1_r1(it);
	VectorXd P_T1_r2(it);
	VectorXd P_T1_r3(it);
	VectorXd P_T1_r4(it);
	VectorXd P_T2_r1(it);
	VectorXd P_T2_r2(it);
	VectorXd P_T2_r3(it);
	VectorXd P_T2_r4(it);
	VectorXd P_T3_r1(it);
	VectorXd P_T3_r2(it);
	VectorXd P_T3_r3(it);
	VectorXd P_T3_r4(it);
	VectorXd P_T4_r1(it);
	VectorXd P_T4_r2(it);
	VectorXd P_T4_r3(it);
	VectorXd P_T4_r4(it);
		
	//vectors of values after burn in and thinning
	//(for use in working out SD of individual NMAs)
	VectorXd d_0vec((NG-burn)/thin);
	VectorXd d_1vec((NG-burn)/thin);
	VectorXd d_2vec((NG-burn)/thin);
	VectorXd d_3vec((NG-burn)/thin);
	VectorXd d_4vec((NG-burn)/thin);
	VectorXd d_5vec((NG-burn)/thin);
	VectorXd tau_vec((NG-burn)/thin);

	//create files to store results
	//Summary file: mean and SD of estimated parameter values
	ofstream filesum; 	
	filesum.open("/path/Summary.txt");

	//treatment specific bias on treatment effect for each treatment
	ofstream file_avbias;
	file_avbias.open("/path/Av_dBias.txt");

	//treatment specific SD on treatment effect for each treatment
	ofstream file_avSD;
	file_avSD.open("/path/Av_dSD.txt");

	//Rank Prob Biases: Mean and SD of bias on the prob that each treatment has each rank
	ofstream fileP;		
	fileP.open("/path/Bias_RankProb.txt");

	//Number of studies per comparison
	ofstream fileN;		
	fileN.open("/path/N.txt");

	//Number of studies per treatment
	ofstream fileNT;	
	fileNT.open("/path/NT.txt");

	//Network values (h^2, \Bar{k}, h^2/\Bar{k}^2, total rank prob bias, total SUCRA bias) 
	ofstream fileNET;	
	fileNET.open("/path/Network_Vals.txt");

	//Bias on SUCRA_alpha alpha=1,2,3,4
	ofstream file_SUCRA;	
	file_SUCRA.open("/path/Bias_SUCRA.txt");
	
	
	//Random Number Generators and Seeds:******************************************************
	//For each parameter to be sampled...
	//... set the seed using PCG library (extra level of randomisation)...
	//... and define the pcg random number engine to be used

	//SIMULATED `DATA':
	//simulated delta (trial specific relative treatment effect)
	// Seed with a real random value
    	pcg_extras::seed_seq_from<std::random_device> seed_delta_sim;
  	// Make a random number engine 
   	pcg32 rng_delta_sim(seed_delta_sim);

	//simulated p values (absolute treatment effect)
	// Seed with a real random value
    	pcg_extras::seed_seq_from<std::random_device> seed_p;
  	// Make a random number engine 
   	pcg32 rng_p(seed_p);

	//simulated r values (sampled events)
	// Seed with a real random value
    	pcg_extras::seed_seq_from<std::random_device> seed_r_sim;
  	// Make a random number engine 
   	pcg32 rng_r_sim(seed_r_sim);


	//NMA:
	//initial value of delta
	// Seed with a real random value
    	pcg_extras::seed_seq_from<std::random_device> seed_delta_0;
  	// Make a random number engine 
   	pcg32 rng_delta_0(seed_delta_0);

	//Gibbs step propose next delta value:
	// Seed with a real random value
    	pcg_extras::seed_seq_from<std::random_device> seed_delta;
  	// Make a random number engine 
   	pcg32 rng_delta(seed_delta);

	//Gibbs step propose next b value:
	// Seed with a real random value
    	pcg_extras::seed_seq_from<std::random_device> seed_b;
  	// Make a random number engine 
   	pcg32 rng_b(seed_b);

	//Gibbs step propose next tau value:
	// Seed with a real random value
    	pcg_extras::seed_seq_from<std::random_device> seed_tau;
  	// Make a random number engine 
   	pcg32 rng_tau(seed_tau);

	//Gibbs step propose next d value:
	// Seed with a real random value
    	pcg_extras::seed_seq_from<std::random_device> seed_d;
  	// Make a random number engine 
   	pcg32 rng_d(seed_d);
	//*****************************************************************************************


	//Iterate over different realisations of simulated data************************************
	for(int x=0; x<it; x++){
		

		//SIMULATE event data**************************************************************
		
		//pick binomial events (vector r) using real parameter values and treat_net
		VectorXi r(a_sum);
		
		for (int s = 0; s < n_s; s++) {		//for each study 
			
			//pick simulated delta values delta_sim(s) from normal distributions N(d_real,tau_real)
			//length of vector delta_sim(s) should be a(s)-1
			VectorXd delta_sim(a(s) - 1);
			
			//need to find the correct d_real to use
			//find the sum over a so that it goes from 0 to a(0)+...+a(n_s-1)
			int sum_at0 = 0;
			for (int j = 0; j < s; j++) {
				sum_at0 = sum_at0 + a(j);
			}
			
			//create vector of d_real for this study (the real values of d_ab for the treatments in this study)
			VectorXd d_reali(a(s) - 1);
			//if baseline treatment in this study is the global baseline - d is just the basic parameters for the treatments in that study
			if (treat_net(sum_at0) == 1) {
				for (int z = 0;z < a(s)-1;z++) {
					d_reali(z) = d_basic_real(treat_net(sum_at0 + z + 1) - 2);
				}
			}
			//if study baseline isn't global baseline then we have to use consistency equations d_ab = d_1b - d_1a
			else {
				for (int y = 0;y < a(s)-1;y++) {
					d_reali(y) = d_basic_real(treat_net(sum_at0 + y + 1) - 2) - d_basic_real(treat_net(sum_at0) - 2);
				}
			}
			
			
			//now pick delta_sim_i from d_real_i 
			//two arm:-----------------------------------------------------------------
			//univariate normal distribution N(d_real, tau)
			if((a(s)-1)==1){
				normal_distribution<double> norm_dist_delta_sim(d_reali(0), tau_real);
				delta_sim(0) = norm_dist_delta_sim(rng_delta_sim);
				
			}
			//multi-arm:---------------------------------------------------------------
			//multivariate normal distribution N(d_real_vec, omega)
			else{
				//mean is just d_reali

				//construct covariance Omega_s = tau_real^2 * C_s
				//create C_s 
				MatrixXd C_s(a(s) - 1, a(s) - 1); //dimensions 
				//diagonal elements of C = 1
				C_s = MatrixXd::Identity(a(s) - 1, a(s) - 1);
				//Off diagonal elemts = 1/2
				for (int r = 0;r < C_s.rows();r++) {
					for (int c = 0;c < C_s.cols();c++) {
						if (C_s(r, c) != 1) {
							C_s(r, c) = 0.5;
						}
						else {
							C_s(r, c) = C_s(r, c);
						}
					}
				}
				

				//Covariance matrix Omega_s = tau_real^2 * C_s
				MatrixXd Covar = pow(tau_real, 2.0)*C_s;

				//now sample deltas using MVN 
				normal_random_variable sample(d_reali,Covar);
				delta_sim = sample();

			}
			
			
			//DATA GENERATING MODEL****************************************************
			//Comment out the two methods not in use:

			//Uniform:-----------------------------------------------------------------
			/*
			//calculate p1 from uniform(0,1):
			uniform_real_distribution<double> uni_distribution_p(0.0, 1.0);
			
			VectorXd p_sim(a(s));
			p_sim(0) = uni_distribution_p(rng_p);//----------------------------------*/
			
			//Normal:------------------------------------------------------------------
			//calculate p1 from Normal(0.5,0.2):
			normal_distribution<double> norm_distribution_p(0.5, 0.2);
			
			VectorXd p_sim(a(s));
			p_sim(0)=-1.0;
			while(p_sim(0)<=0.0 || p_sim(0)>=1.0){ //clip probability dist at 0 and 1
				p_sim(0) = norm_distribution_p(rng_p);
			}//----------------------------------------------------------------------*/
			
			//Euclidean----------------------------------------------------------------
			/*
			//calc p1 from minimisation
			VectorXd p_sim(a(s));
			vector<double> Func(999);
			for(int f=1; f<=999; f++){
				double p = f*0.001;
				Func[f-1] = pow((p-0.5),2.0);
				for(int g=0; g<(a(s)-1); g++){
					Func[f-1] = Func[f-1] + pow((p*exp(delta_sim(g))/(1 - p + p*exp(delta_sim(g)))) - 0.5, 2.0);
				}
			}
			//make a copy of func to be sorted
			vector<double> f_sort(999);
			f_sort = Func;
			sort(f_sort.begin(),f_sort.end());
			double min = f_sort[0];

			auto position = find(Func.begin(), Func.end(), min);
			int pos = distance(Func.begin(), position);
			p_sim(0) = (pos+1)*0.001;//----------------------------------------------*/

			//calc pa from p1---------------------------------------------------------
			for (int t = 1;t < a(s);t++) {
				p_sim(t) = (p_sim(0)*exp(delta_sim(t - 1))) / (1 - p_sim(0) + p_sim(0)*exp(delta_sim(t - 1)));
			}
			
			//*************************************************************************

			//pick r's from binomial distribution Bin(p,n)
			
			for (int t = 0;t < a(s);t++) {
				binomial_distribution<int> bin_dist_r_sim(n(sum_at0+t), p_sim(t));
				r(sum_at0 + t) = bin_dist_r_sim(rng_r_sim);
			}
		}
		//*********************************************************************************

	
		//NMA STARTS HERE: ****************************************************************
		
		//Initialise values:***************************************************************

		//1. Initialise vector of basic parameters d_basic (choose starting values)
		//DO NOT choose any d_basic components to be equal to each other or = 0.0 (or creating delta won't work)
		VectorXd d_basic(n_t - 1);
		d_basic << 0.101, 0.202, 0.303; 

		//create vector d from d_basic and treat_net
		VectorXd d(a_sum);
		//---------------------------------------------------------------------------------
		for (int t = 0;t < n_s;t++) {
			//find the sum over a so that it goes from 0 to a(0)+...+a(n_s-1)
			int sum_at0 = 0;
			for (int j = 0; j < t; j++) {
				sum_at0 = sum_at0 + a(j);
			}
			
			d(sum_at0) = 0.0;		//d_AA is always 0 

			//if study t includes global baseline (1) - just basic parameters
			if (treat_net(sum_at0) == 1) {
				for (int z = 1;z < a(t);z++) {
					d(sum_at0 + z) = d_basic(treat_net(sum_at0 + z) - 2);
				}
			}
			//if study t does not include global baseline - use consistency equations
			else {
				for (int y = 1;y < a(t);y++) {
					d(sum_at0 + y) = d_basic(treat_net(sum_at0 + y) - 2)-d_basic(treat_net(sum_at0)-2);
				}
			}

		}
		//---------------------------------------------------------------------------------
		
		//2. Initialise tau (choose any reasonable value - i.e. between 0 and 5)
		double tau = 1.2;

		
		//3. Initialise delta_i from N(d,tau) 
		//We use delta=0 to indicate delta_AA therefore we don't want any of delta components to be exactly zero if they aren't meant to be 
		VectorXd delta(a_sum);
		
		for (int i = 0; i < a_sum; i++) {
			if (d(i) != 0.0) {
				normal_distribution<double> norm_distribution(d(i), tau);
				delta(i) = norm_distribution(rng_delta_0);
			}
			else {
				delta(i) = d(i);
			}
		}
		

		//4. Initialise b vector 
		VectorXd b(n_s);
		for (int i = 0;i < n_s;i++) {
			b(i) = 0.12;
		}
		
		//Sum values that will be used to work out the estimated parameter values
		//estimated parameter value = mean over Gibbs iterations
		double sum_d0 = 0.0;
		double sum_d1 = 0.0;
		double sum_d2 = 0.0;
		double sum_tau = 0.0;

		//create N_ranki vectors to count number of times each treatment gets each rank
		VectorXi N_rankT1(n_t);
		VectorXi N_rankT2(n_t);
		VectorXi N_rankT3(n_t);
		VectorXi N_rankT4(n_t);

		//5. Initialise all values of N_rank vectors to 0
		for(int r=0; r<n_t; r++){
			N_rankT1(r)=0;
			N_rankT2(r)=0;
			N_rankT3(r)=0;
			N_rankT4(r)=0;
		}
		//*********************************************************************************
		
		//Gibbs Iterations*****************************************************************
		//At each Gibbs iteration we pick a proposed new parameter value conditional on the current values of all parameters and choose whether to accept or reject this proposed value based on the value of the probability distributions at the current and proposed value. We do this for each model parameter in turn. 
		for (int t = 0; t < NG; t++) {

			//delta step***************************************************************
			//define a vector of b's using a and b that has same dimensions as delta, r and n (for use in the selection 
			VectorXd b_extra(a_sum);
			for (int i = 0; i < n_s;i++) {
				//find starting point 
				int sum_a_s = 0;
				for (int s = 0;s < i; s++) {
					sum_a_s = sum_a_s + a(s);
				}

				//find final point (+1)
				int sum_a_f = 0;
				for (int f = 0;f <= i; f++) {
					sum_a_f = sum_a_f + a(f);
				}

				for (int j = sum_a_s; j < sum_a_f;j++) {
					b_extra(j) = b(i);
				}

			}
			
		
			int trial = 0;
			
			for (int i = 0; i < a_sum;i++) {
				
				//we only want to change delta(i) not equal to 0
				if (delta(i) != 0.0) {
					//pick a proposed value for delta[i] from N(delta_i,v) 
					normal_distribution<double> norm_dist_delta(delta(i), sd_prop_delta);
					double delta_i_prop = norm_dist_delta(rng_delta);
					
					//work out log(probability density) at current value
					//using delta_vec with delta(i)
					double ln_density_current = ln_delta_density(trial - 1, delta(i), b_extra(i), r(i), n(i), tau, delta, d, a);

					//hold onto current value
					double delta_hold = delta(i);
					//need to edit delta(i) in delta_vec: change delta(i) to delta_prop
					delta(i) = delta_i_prop;
					//work out log(probability density) at proposed value
					//using delta_vec with delta_prop
					double ln_density_prop = ln_delta_density(trial - 1, delta_i_prop, b_extra(i), r(i), n(i), tau, delta, d, a);
					//change delta back to current value for MH step
					delta(i) = delta_hold;
					//now accept or reject using MH function to get new value of delta[i]
					//this will choose between delta(i) and delta(prop)
					delta(i) = MH_nextvalue(ln_density_current, ln_density_prop, delta(i), delta_i_prop);	

				}
				else {
					trial = trial + 1;
				}
			}
			//*************************************************************************

			//b step*******************************************************************
			//do MH step for b_i (pick new b)
			
			
			for (int i = 0; i < n_s;i++) {
				//pick a proposed value for b[i] from N(b_i,v) 
				normal_distribution<double> norm_dist_b(b(i), sd_prop_b);
				double b_i_prop = norm_dist_b(rng_b);
				//work out density of each value
				double ln_density_current_b = ln_b_density(i + 1, b(i), s_b, r, n, a, delta);
				double ln_density_prop_b = ln_b_density(i + 1, b_i_prop, s_b, r, n, a, delta);
				//now accept or reject using MH function to get new value of b[i]
				b(i) = MH_nextvalue(ln_density_current_b, ln_density_prop_b, b(i), b_i_prop);
	
			}

			//*************************************************************************
			
			//tau step*****************************************************************
			//do MH step for tau (pick new tau)
			//pick a proposed value for tau from N(tau,v) 
			normal_distribution<double> norm_dist_tau(tau, sd_prop_tau);
			double tau_prop = norm_dist_tau(rng_tau);
			//now accept or reject using MH function to get new value of tau
			tau = MH_nextvalue_tau(tau, tau_prop, d, n_s, delta, a);

			//*************************************************************************

			//d step*******************************************************************
			for (int i = 0; i < n_t - 1; i++) {
				//pick a proposed value for d_basic[i] from N(d_basic_i,v) 
				normal_distribution<double> norm_dist_d(d_basic(i), sd_prop_d);
				double d_i_prop = norm_dist_d(rng_d);

				//work out density of each value
				//vector d includes d(i) current
				double ln_density_current_d = ln_d_density(d_basic(i), tau, d, delta, n_s, a, s_d);

				//hold on to current value
				double d_hold = d_basic(i);

				//need to change vector d to include d_prop for ln_density_prop
				//we work out vector d from d_basic as before
				d_basic(i) = d_i_prop;
				//-----------------------------------------------------------------
				for (int tD = 0;tD < n_s;tD++) {
					//find the sum over a so that it goes from 0 to a(0)+...+a(n_s-1)
					int sum_at0 = 0;
					for (int j = 0; j < tD; j++) {
						sum_at0 = sum_at0 + a(j);
					}

					//d_AA is always 0 
					d(sum_at0) = 0;
					//if study t includes global baseline (1) - just basic parameters
					if (treat_net(sum_at0) == 1) {
						for (int z = 1;z < a(tD);z++) {
							d(sum_at0 + z) = d_basic(treat_net(sum_at0 + z) - 2);
						}
					}
					//if study t does not include global baseline - use consistency equations
					else {
						for (int y = 1;y < a(tD);y++) {
							d(sum_at0 + y) = d_basic(treat_net(sum_at0 + y) - 2)-d_basic(treat_net(sum_at0)-2);
						}
					}

				}
				//-----------------------------------------------------------------
				
				//vector d includes d_prop
				double ln_density_prop_d = ln_d_density(d_i_prop, tau, d, delta, n_s, a, s_d);
				//change back to d current before MH step
				d_basic(i) = d_hold;
								
				//now accept or reject using MH function to get new value of d[i]
				//this will change d_basic(i) back to d_basic(i) if necessary
				d_basic(i) = MH_nextvalue(ln_density_current_d, ln_density_prop_d, d_basic(i), d_i_prop);

				//now redefine d (given d_basic(i))--------------------------------
				for (int tD = 0;tD < n_s;tD++) {
					//find the sum over a so that it goes from 0 to a(0)+...+a(n_s-1)
					int sum_at0 = 0;
					for (int j = 0; j < tD; j++) {
						sum_at0 = sum_at0 + a(j);
					}
					
					//d_AA is always 0 
					d(sum_at0) = 0;
					//if study t includes global baseline (1) - just basic parameters
					if (treat_net(sum_at0) == 1) {
						for (int z = 1;z < a(tD);z++) {
							d(sum_at0 + z) = d_basic(treat_net(sum_at0 + z) - 2);
						}
					}
					//if study t does not include global baseline - use consistency equations
					else {
						for (int y = 1;y < a(tD);y++) {
							d(sum_at0 + y) = d_basic(treat_net(sum_at0 + y) - 2)-d_basic(treat_net(sum_at0)-2);
						}
					}

				}
				//-----------------------------------------------------------------
				
				
			}
			
			//Now ensure burn in and thinning 
			//only keep iterations after `burn' iterations 
			//After burn in - only keep every 10 iterations (where thin=10)
			if (t >= burn && (t+1) % thin == 0) { 
				
				sum_d0 = sum_d0 + d_basic(0);
				sum_d1 = sum_d1 + d_basic(1);
				sum_d2 = sum_d2 + d_basic(2);
				sum_tau = sum_tau + tau;
				
				//assign values to vectors (to work out SD)
				//Edit values to account for burn in and thinning
				d_0vec((t-5009)/10)=d_basic(0);
				d_1vec((t-5009)/10)=d_basic(1);
				d_2vec((t-5009)/10)=d_basic(2);
				d_3vec((t-5009)/10)=d_basic(1) - d_basic(0);
				d_4vec((t-5009)/10)=d_basic(2) - d_basic(0);
				d_5vec((t-5009)/10)=d_basic(2) - d_basic(1);
				tau_vec((t-5009)/10)=tau;
				
				//ranking: rank treatments based on d values at this iteration
				//create a vector with 0 as first entry and d_basic as rest
				vector<double> d_rank(n_t);
				d_rank[0]=0.0;
				for(int i=1;i<n_t;i++){
					d_rank[i]=d_basic(i-1);
				}
				//create a copy of d_rank to be sorted
				vector<double> d_sort(n_t);
				for(int i=0; i<n_t; i++){
					d_sort[i] = d_rank[i];
				}
				//now sort the copy from smallest to largest
				sort(d_sort.begin(), d_sort.end());
				//compare d_rank and d_sort to give ranks to treatments
				//create vector of ranks 
				VectorXi R(n_t);
				for(int i=0; i<n_t; i++){
					for(int j=0; j<n_t; j++){
						if(d_rank[i]==d_sort[j]){R(i)=j+1;}
					}
				}
				
				//using vector R - update counts in N_ranki vectors
				for(int i=0; i<n_t; i++){ //i=0 checks rank1, i=1 checks rank 2 etc...
					if( R(0)==i+1 ){ N_rankT1(i)=N_rankT1(i)+1; } //R(0) checks T1
					else if( R(1)==i+1 ){ N_rankT2(i)=N_rankT2(i)+1; } //R(1) checks T2
					else if( R(2)==i+1 ){ N_rankT3(i)=N_rankT3(i)+1; } //R(2) checks T3
					else if( R(3)==i+1 ){ N_rankT4(i)=N_rankT4(i)+1; } //R(3) checks T4
				}
				

			}
			else {
				sum_d0 = sum_d0;
				sum_d1 = sum_d1;
				sum_d2 = sum_d2;
				sum_tau = sum_tau;
				
			}


		}
		//END OF NMA GIBBS ITERATIONS******************************************************

		//Probability Ranks****************************************************************
		//calculate BIAS on P(R) for each treatment and each rank R and assign to vectors using N_RankTi	
		VectorXd P_rankT1(n_t);
		VectorXd P_rankT2(n_t);
		VectorXd P_rankT3(n_t);
		VectorXd P_rankT4(n_t);
		//This is for equally effective treatments and networks with 4 treatments (true P(r)=1/4 for all 4 treatments)
		//Edit this for non-equally effective treatments and/or networks with more/fewer treatments
		for(int p=0; p<n_t; p++){
			P_rankT1(p)=(N_rankT1(p)/N_new)-0.25; //bias on probability T1 has rank p+1
			P_rankT2(p)=(N_rankT2(p)/N_new)-0.25; //bias on prob that T2 has rank p+1
			P_rankT3(p)=(N_rankT3(p)/N_new)-0.25; //bias on prob that T3 has rank p+1
			P_rankT4(p)=(N_rankT4(p)/N_new)-0.25; //bias on prob that T4 has rank p+1
		}
		
		//assign to simulation vectors (to work out mean and sd over simulation iterations)
		P_T1_r1(x) = P_rankT1(0);
		P_T1_r2(x) = P_rankT1(1);
		P_T1_r3(x) = P_rankT1(2);
		P_T1_r4(x) = P_rankT1(3);
		P_T2_r1(x) = P_rankT2(0);
		P_T2_r2(x) = P_rankT2(1);
		P_T2_r3(x) = P_rankT2(2);
		P_T2_r4(x) = P_rankT2(3);
		P_T3_r1(x) = P_rankT3(0);
		P_T3_r2(x) = P_rankT3(1);
		P_T3_r3(x) = P_rankT3(2);
		P_T3_r4(x) = P_rankT3(3);
		P_T4_r1(x) = P_rankT4(0);
		P_T4_r2(x) = P_rankT4(1);
		P_T4_r3(x) = P_rankT4(2);
		P_T4_r4(x) = P_rankT4(3);

		//*********************************************************************************
		
		//Model Parameter Estimates********************************************************
		//Work out mean of ds and tau over Gibbs iterations to get NMA estimates of model parameter values (\Tilde{d}_ab and \Tilde{tau})		
		double mean_d0 = sum_d0 / N_new;
		double mean_d1 = sum_d1 / N_new;
		double mean_d2 = sum_d2 / N_new;
		double mean_d3 = mean_d1 - mean_d0;
		double mean_d4 = mean_d2 - mean_d0;
		double mean_d5 = mean_d2 - mean_d1;
		double mean_tau = sum_tau / N_new;
		
		//for working out standard deviations 
		//(sum over the difference between estimated and mean values - squared)
		double sum_diff0 = 0.0;
		double sum_diff1 = 0.0;
		double sum_diff2 = 0.0;	
		double sum_diff3 = 0.0;
		double sum_diff4 = 0.0;
		double sum_diff5 = 0.0;
		double sum_difftau = 0.0;				

		for(int c=0; c<(NG-burn)/thin; c++){
			sum_diff0 = sum_diff0 + pow((d_0vec(c)-mean_d0),2.0);
			sum_diff1 = sum_diff1 + pow((d_1vec(c)-mean_d1),2.0);
			sum_diff2 = sum_diff2 + pow((d_2vec(c)-mean_d2),2.0);
			sum_diff3 = sum_diff3 + pow((d_3vec(c)-mean_d3),2.0);
			sum_diff4 = sum_diff4 + pow((d_4vec(c)-mean_d4),2.0);
			sum_diff5 = sum_diff5 + pow((d_5vec(c)-mean_d5),2.0);
			sum_difftau = sum_difftau + pow((tau_vec(c)-mean_tau),2.0);
		}
		double SD_d0 = pow((sum_diff0/(N_new-1.0)),0.5);
		double SD_d1 = pow((sum_diff1/(N_new-1.0)),0.5);
		double SD_d2 = pow((sum_diff2/(N_new-1.0)),0.5);
		double SD_d3 = pow((sum_diff3/(N_new-1.0)),0.5);
		double SD_d4 = pow((sum_diff4/(N_new-1.0)),0.5);
		double SD_d5 = pow((sum_diff5/(N_new-1.0)),0.5);
		double SD_tau = pow((sum_difftau/(N_new-1.0)),0.5);
	
		//assign each mean and sd value to the appropriate vector
		//so we can work out mean of estimates/SDs over the simulation iterations
		vec_mean_d0(x) = mean_d0;
		vec_sd_d0(x) = SD_d0;
		vec_mean_d1(x) = mean_d1;
		vec_sd_d1(x) = SD_d1;
		vec_mean_d2(x) = mean_d2;
		vec_sd_d2(x) = SD_d2;
		vec_mean_d3(x) = mean_d3;
		vec_sd_d3(x) = SD_d3;
		vec_mean_d4(x) = mean_d4;
		vec_sd_d4(x) = SD_d4;
		vec_mean_d5(x) = mean_d5;
		vec_sd_d5(x) = SD_d5;
		vec_mean_tau(x) = mean_tau;
		vec_sd_tau(x) = SD_tau;
		//*********************************************************************************
	}
	//END OF SIMULATION/NMA ITERATIONS*********************************************************
	
	//Rank Probabilities***********************************************************************
	//work out mean of bias on P_a(r) over simulation iterations
	double sumP_T1_r1 = 0.0;
	double sumP_T1_r2 = 0.0;
	double sumP_T1_r3 = 0.0;
	double sumP_T1_r4 = 0.0;
	double sumP_T2_r1 = 0.0;
	double sumP_T2_r2 = 0.0;
	double sumP_T2_r3 = 0.0;
	double sumP_T2_r4 = 0.0;
	double sumP_T3_r1 = 0.0;
	double sumP_T3_r2 = 0.0;
	double sumP_T3_r3 = 0.0;
	double sumP_T3_r4 = 0.0;
	double sumP_T4_r1 = 0.0;
	double sumP_T4_r2 = 0.0;
	double sumP_T4_r3 = 0.0;
	double sumP_T4_r4 = 0.0;
	for(int c=0; c<it; c++){
		sumP_T1_r1 = sumP_T1_r1 + P_T1_r1(c);
		sumP_T1_r2 = sumP_T1_r2 + P_T1_r2(c);
		sumP_T1_r3 = sumP_T1_r3 + P_T1_r3(c);
		sumP_T1_r4 = sumP_T1_r4 + P_T1_r4(c);

		sumP_T2_r1 = sumP_T2_r1 + P_T2_r1(c);
		sumP_T2_r2 = sumP_T2_r2 + P_T2_r2(c);
		sumP_T2_r3 = sumP_T2_r3 + P_T2_r3(c);
		sumP_T2_r4 = sumP_T2_r4 + P_T2_r4(c);

		sumP_T3_r1 = sumP_T3_r1 + P_T3_r1(c);
		sumP_T3_r2 = sumP_T3_r2 + P_T3_r2(c);
		sumP_T3_r3 = sumP_T3_r3 + P_T3_r3(c);
		sumP_T3_r4 = sumP_T3_r4 + P_T3_r4(c);

		sumP_T4_r1 = sumP_T4_r1 + P_T4_r1(c);
		sumP_T4_r2 = sumP_T4_r2 + P_T4_r2(c);
		sumP_T4_r3 = sumP_T4_r3 + P_T4_r3(c);
		sumP_T4_r4 = sumP_T4_r4 + P_T4_r4(c);

	}
	
	double meanP_T1_r1 = sumP_T1_r1/dit;
	double meanP_T1_r2 = sumP_T1_r2/dit;
	double meanP_T1_r3 = sumP_T1_r3/dit;
	double meanP_T1_r4 = sumP_T1_r4/dit;

	double meanP_T2_r1 = sumP_T2_r1/dit;
	double meanP_T2_r2 = sumP_T2_r2/dit;
	double meanP_T2_r3 = sumP_T2_r3/dit;
	double meanP_T2_r4 = sumP_T2_r4/dit;

	double meanP_T3_r1 = sumP_T3_r1/dit;
	double meanP_T3_r2 = sumP_T3_r2/dit;
	double meanP_T3_r3 = sumP_T3_r3/dit;
	double meanP_T3_r4 = sumP_T3_r4/dit;

	double meanP_T4_r1 = sumP_T4_r1/dit;
	double meanP_T4_r2 = sumP_T4_r2/dit;
	double meanP_T4_r3 = sumP_T4_r3/dit;
	double meanP_T4_r4 = sumP_T4_r4/dit;

	//work out SD of bias on P_a(r) over simulation iterations
	double sumdiffP_T1_r1 = 0.0;
	double sumdiffP_T1_r2 = 0.0;
	double sumdiffP_T1_r3 = 0.0;
	double sumdiffP_T1_r4 = 0.0;

	double sumdiffP_T2_r1 = 0.0;
	double sumdiffP_T2_r2 = 0.0;
	double sumdiffP_T2_r3 = 0.0;
	double sumdiffP_T2_r4 = 0.0;

	double sumdiffP_T3_r1 = 0.0;
	double sumdiffP_T3_r2 = 0.0;
	double sumdiffP_T3_r3 = 0.0;
	double sumdiffP_T3_r4 = 0.0;

	double sumdiffP_T4_r1 = 0.0;
	double sumdiffP_T4_r2 = 0.0;
	double sumdiffP_T4_r3 = 0.0;
	double sumdiffP_T4_r4 = 0.0;

	for(int c=0; c<it; c++){
		sumdiffP_T1_r1 = sumdiffP_T1_r1 + pow((P_T1_r1(c)-meanP_T1_r1),2.0);
		sumdiffP_T1_r2 = sumdiffP_T1_r2 + pow((P_T1_r2(c)-meanP_T1_r2),2.0);
		sumdiffP_T1_r3 = sumdiffP_T1_r3 + pow((P_T1_r3(c)-meanP_T1_r3),2.0);
		sumdiffP_T1_r4 = sumdiffP_T1_r4 + pow((P_T1_r4(c)-meanP_T1_r4),2.0);

		sumdiffP_T2_r1 = sumdiffP_T2_r1 + pow((P_T2_r1(c)-meanP_T2_r1),2.0);
		sumdiffP_T2_r2 = sumdiffP_T2_r2 + pow((P_T2_r2(c)-meanP_T2_r2),2.0);
		sumdiffP_T2_r3 = sumdiffP_T2_r3 + pow((P_T2_r3(c)-meanP_T2_r3),2.0);
		sumdiffP_T2_r4 = sumdiffP_T2_r4 + pow((P_T2_r4(c)-meanP_T2_r4),2.0);

		sumdiffP_T3_r1 = sumdiffP_T3_r1 + pow((P_T3_r1(c)-meanP_T3_r1),2.0);
		sumdiffP_T3_r2 = sumdiffP_T3_r2 + pow((P_T3_r2(c)-meanP_T3_r2),2.0);
		sumdiffP_T3_r3 = sumdiffP_T3_r3 + pow((P_T3_r3(c)-meanP_T3_r3),2.0);
		sumdiffP_T3_r4 = sumdiffP_T3_r4 + pow((P_T3_r4(c)-meanP_T3_r4),2.0);
		
		sumdiffP_T4_r1 = sumdiffP_T4_r1 + pow((P_T4_r1(c)-meanP_T4_r1),2.0);
		sumdiffP_T4_r2 = sumdiffP_T4_r2 + pow((P_T4_r2(c)-meanP_T4_r2),2.0);
		sumdiffP_T4_r3 = sumdiffP_T4_r3 + pow((P_T4_r3(c)-meanP_T4_r3),2.0);
		sumdiffP_T4_r4 = sumdiffP_T4_r4 + pow((P_T4_r4(c)-meanP_T4_r4),2.0);
	}
	double SDP_T1_r1 = pow((sumdiffP_T1_r1/(dit-1.0)),0.5);
	double SDP_T1_r2 = pow((sumdiffP_T1_r2/(dit-1.0)),0.5);
	double SDP_T1_r3 = pow((sumdiffP_T1_r3/(dit-1.0)),0.5);
	double SDP_T1_r4 = pow((sumdiffP_T1_r4/(dit-1.0)),0.5);

	double SDP_T2_r1 = pow((sumdiffP_T2_r1/(dit-1.0)),0.5);
	double SDP_T2_r2 = pow((sumdiffP_T2_r2/(dit-1.0)),0.5);
	double SDP_T2_r3 = pow((sumdiffP_T2_r3/(dit-1.0)),0.5);
	double SDP_T2_r4 = pow((sumdiffP_T2_r4/(dit-1.0)),0.5);

	double SDP_T3_r1 = pow((sumdiffP_T3_r1/(dit-1.0)),0.5);
	double SDP_T3_r2 = pow((sumdiffP_T3_r2/(dit-1.0)),0.5);
	double SDP_T3_r3 = pow((sumdiffP_T3_r3/(dit-1.0)),0.5);
	double SDP_T3_r4 = pow((sumdiffP_T3_r4/(dit-1.0)),0.5);

	double SDP_T4_r1 = pow((sumdiffP_T4_r1/(dit-1.0)),0.5);
	double SDP_T4_r2 = pow((sumdiffP_T4_r2/(dit-1.0)),0.5);
	double SDP_T4_r3 = pow((sumdiffP_T4_r3/(dit-1.0)),0.5);
	double SDP_T4_r4 = pow((sumdiffP_T4_r4/(dit-1.0)),0.5);
	//*****************************************************************************************

	//Model Parameter Estimates****************************************************************
	//using vectors of means and sds - work out the mean/sd of means and sd's

	//work out mean of estimated parameter values over simulation iterations
	double sum0m = 0.0;
	double sum1m = 0.0;
	double sum2m = 0.0;
	double sumTm = 0.0;
	for(int c=0; c<it; c++){
		sum0m=sum0m+vec_mean_d0(c);
		sum1m=sum1m+vec_mean_d1(c);
		sum2m=sum2m+vec_mean_d2(c);
		sumTm=sumTm+vec_mean_tau(c);

	}
	double mean0m = sum0m/dit;
	double mean1m = sum1m/dit;
	double mean2m = sum2m/dit;
	double mean3m = mean1m - mean0m;
	double mean4m = mean2m - mean0m;
	double mean5m = mean2m - mean1m;
	double meanTm = sumTm/dit;

	//work out SD of estimated parameter values over simulation iterations
	double sum_diff0m = 0.0;
	double sum_diff1m = 0.0;
	double sum_diff2m = 0.0;
	double sum_diff3m = 0.0;
	double sum_diff4m = 0.0;
	double sum_diff5m = 0.0;
	double sum_diffTm = 0.0;
	for(int c=0; c<it; c++){
		sum_diff0m = sum_diff0m + pow((vec_mean_d0(c)-mean0m),2.0);
		sum_diff1m = sum_diff1m + pow((vec_mean_d1(c)-mean1m),2.0);
		sum_diff2m = sum_diff2m + pow((vec_mean_d2(c)-mean2m),2.0);
		sum_diff3m = sum_diff3m + pow((vec_mean_d3(c)-mean3m),2.0);
		sum_diff4m = sum_diff4m + pow((vec_mean_d4(c)-mean4m),2.0);
		sum_diff5m = sum_diff5m + pow((vec_mean_d5(c)-mean5m),2.0);
		sum_diffTm = sum_diffTm + pow((vec_mean_tau(c)-meanTm),2.0);
	}
	double SD_mean0m = pow((sum_diff0m/(dit-1.0)),0.5);
	double SD_mean1m = pow((sum_diff1m/(dit-1.0)),0.5);
	double SD_mean2m = pow((sum_diff2m/(dit-1.0)),0.5);
	double SD_mean3m = pow((sum_diff3m/(dit-1.0)),0.5);
	double SD_mean4m = pow((sum_diff4m/(dit-1.0)),0.5);
	double SD_mean5m = pow((sum_diff5m/(dit-1.0)),0.5);
	double SD_meanTm = pow((sum_diffTm/(dit-1.0)),0.5);

	//work out mean of SDs
	double sum0sd = 0.0;
	double sum1sd = 0.0;
	double sum2sd = 0.0;
	double sum3sd = 0.0;
	double sum4sd = 0.0;
	double sum5sd = 0.0;
	double sumTsd = 0.0;
	for(int c=0; c<it; c++){
		sum0sd = sum0sd + vec_sd_d0(c);
		sum1sd = sum1sd + vec_sd_d1(c);
		sum2sd = sum2sd + vec_sd_d2(c);
		sum3sd = sum3sd + vec_sd_d3(c);
		sum4sd = sum4sd + vec_sd_d4(c);
		sum5sd = sum5sd + vec_sd_d5(c);
		sumTsd = sumTsd + vec_sd_tau(c);

	}
	double mean0sd = sum0sd/dit;
	double mean1sd = sum1sd/dit;
	double mean2sd = sum2sd/dit;
	double mean3sd = sum3sd/dit;
	double mean4sd = sum4sd/dit;
	double mean5sd = sum5sd/dit;
	double meanTsd = sumTsd/dit;

	//work out SD of SDs

	double sum_diff0sd = 0.0;
	double sum_diff1sd = 0.0;
	double sum_diff2sd = 0.0;
	double sum_diff3sd = 0.0;
	double sum_diff4sd = 0.0;
	double sum_diff5sd = 0.0;
	double sum_diffTsd = 0.0;
	for(int c=0; c<it; c++){
		sum_diff0sd = sum_diff0sd + pow((vec_sd_d0(c)-mean0sd),2.0);
		sum_diff1sd = sum_diff1sd + pow((vec_sd_d1(c)-mean1sd),2.0);
		sum_diff2sd = sum_diff2sd + pow((vec_sd_d2(c)-mean2sd),2.0);
		sum_diff3sd = sum_diff3sd + pow((vec_sd_d3(c)-mean3sd),2.0);
		sum_diff4sd = sum_diff4sd + pow((vec_sd_d4(c)-mean4sd),2.0);
		sum_diff5sd = sum_diff5sd + pow((vec_sd_d5(c)-mean5sd),2.0);
		sum_diffTsd = sum_diffTsd + pow((vec_sd_tau(c)-meanTsd),2.0);

	}
	double SD_mean0sd = pow((sum_diff0sd/(dit-1.0)),0.5);
	double SD_mean1sd = pow((sum_diff1sd/(dit-1.0)),0.5);
	double SD_mean2sd = pow((sum_diff2sd/(dit-1.0)),0.5);
	double SD_mean3sd = pow((sum_diff3sd/(dit-1.0)),0.5);
	double SD_mean4sd = pow((sum_diff4sd/(dit-1.0)),0.5);
	double SD_mean5sd = pow((sum_diff5sd/(dit-1.0)),0.5);
	double SD_meanTsd = pow((sum_diffTsd/(dit-1.0)),0.5);

	//*****************************************************************************************

	//Write to Files***************************************************************************

	//Summary File:----------------------------------------------------------------------------
	//mean of estimated parameter values (\Tilde{d}_ab \Tilde{tau}\) 
	string file_str = to_string(mean0m) + "\n" + to_string(mean1m) + "\n" + to_string(mean2m) + "\n" + to_string(mean3m) + "\n" + to_string(mean4m) + "\n" + to_string(mean5m) + "\n" + to_string(meanTm) + "\n";
	//+ SD of estimated parameter values
	file_str = file_str + to_string(SD_mean0m) + "\n" + to_string(SD_mean1m) + "\n" + to_string(SD_mean2m) + "\n" + to_string(SD_mean3m) + "\n" + to_string(SD_mean4m) + "\n" + to_string(SD_mean5m) + "\n" + to_string(SD_meanTm) + "\n";
	//+ mean of SDs
	file_str = file_str + to_string(mean0sd) + "\n" + to_string(mean1sd) + "\n" + to_string(mean2sd) + "\n" + to_string(mean3sd) + "\n" + to_string(mean4sd) + "\n" + to_string(mean5sd) + "\n" + to_string(meanTsd) + "\n";
	//+SD(SDs)
	file_str = file_str + to_string(SD_mean0sd) + "\n" + to_string(SD_mean1sd) + "\n" + to_string(SD_mean2sd) + "\n" + to_string(SD_mean3sd) + "\n" + to_string(SD_mean4sd) + "\n" + to_string(SD_mean5sd) + "\n" + to_string(SD_meanTsd) + "\n";
	
	//write string to file
	filesum << file_str;	
	//-----------------------------------------------------------------------------------------

	//Average Bias and Average SD on d---------------------------------------------------------
	VectorXd avBias(n_t);
	avBias(0) = (mean0m + mean1m + mean2m)/3.0;
	avBias(1) = (- mean0m + mean3m + mean4m)/3.0;
	avBias(2) = (- mean1m - mean3m + mean5m)/3.0;
	avBias(3) = (- mean2m - mean4m - mean5m)/3.0;
	for(int i=0; i<n_t; i++){
		file_avbias << to_string(avBias(i)) << "\n";
	}

	VectorXd avSD(n_t);
	avSD(0) = (SD_mean0m + SD_mean1m + SD_mean2m)/3.0;
	avSD(1) = (SD_mean0m + SD_mean3m + SD_mean4m)/3.0;
	avSD(2) = (SD_mean1m + SD_mean3m + SD_mean5m)/3.0;
	avSD(3) = (SD_mean2m + SD_mean4m + SD_mean5m)/3.0;
	for(int i=0; i<n_t; i++){
		file_avSD << to_string(avSD(i)) << "\n";
	}
	//-----------------------------------------------------------------------------------------

	//Probability Ranks:-----------------------------------------------------------------------
	//Treatment 1: Mean(Bias P_1(r)) and SD(Bias P_1(r)) {r=1,2,3,4}
	fileP << to_string(meanP_T1_r1) << "\n" << to_string(meanP_T1_r2) << "\n" << to_string(meanP_T1_r3) << "\n" << to_string(meanP_T1_r4) << "\n" << to_string(SDP_T1_r1) << "\n" << to_string(SDP_T1_r2) << "\n" << to_string(SDP_T1_r3) << "\n" << to_string(SDP_T1_r4) << "\n";
	//+ Treatment 2: Mean(Bias P_2(r)) and SD(Bias P_2(r)) {r=1,2,3,4}
	fileP << to_string(meanP_T2_r1) << "\n" << to_string(meanP_T2_r2) << "\n" << to_string(meanP_T2_r3) << "\n" << to_string(meanP_T2_r4) << "\n" << to_string(SDP_T2_r1) << "\n" << to_string(SDP_T2_r2) << "\n" << to_string(SDP_T2_r3) << "\n" << to_string(SDP_T2_r4) << "\n";
	//+ Treatment 3: Mean(Bias P_3(r)) and SD(Bias P_3(r)) {r=1,2,3,4}
	fileP << to_string(meanP_T3_r1) << "\n" << to_string(meanP_T3_r2) << "\n" << to_string(meanP_T3_r3) << "\n" << to_string(meanP_T3_r4) << "\n" << to_string(SDP_T3_r1) << "\n" << to_string(SDP_T3_r2) << "\n" << to_string(SDP_T3_r3) << "\n" << to_string(SDP_T3_r4) << "\n";
	//+ Treatment 4: Mean(Bias P_4(r)) and SD(Bias P_4(r)) {r=1,2,3,4}
	fileP << to_string(meanP_T4_r1) << "\n" << to_string(meanP_T4_r2) << "\n" << to_string(meanP_T4_r3) << "\n" << to_string(meanP_T4_r4) << "\n" << to_string(SDP_T4_r1) << "\n" << to_string(SDP_T4_r2) << "\n" << to_string(SDP_T4_r3) << "\n" << to_string(SDP_T4_r4);
	//-----------------------------------------------------------------------------------------

	//write N vector to file-------------------------------------------------------------------
	//Number of studies per comparison
	for (int n=0;n<l;n++){
		fileN << to_string(N(n)) << "\n";
	}

	//Number of studies per treatment
	VectorXd NT(n_t);
	NT(0) = N(0)+N(1)+N(2);
	NT(1) = N(0)+N(3)+N(4);
	NT(2) = N(1)+N(3)+N(5);
	NT(3) = N(2)+N(4)+N(5);
	fileNT << to_string(N(0)+N(1)+N(2)) << "\n" << to_string(N(0)+N(3)+N(4)) << "\n" << to_string(N(1)+N(3)+N(5)) << "\n" << to_string(N(2)+N(4)+N(5));
	//-----------------------------------------------------------------------------------------
	

	//Network Values---------------------------------------------------------------------------
	
	//total rank probability bias 
	//Sum over ranks (r) and treatments (a) of absolute bias on P_a(r)	
	double TMB = abs(meanP_T1_r1) + abs(meanP_T2_r1) + abs(meanP_T3_r1) + abs(meanP_T4_r1) + abs(meanP_T1_r2) + abs(meanP_T2_r2) + abs(meanP_T3_r2) + abs(meanP_T4_r2) + abs(meanP_T1_r3) + abs(meanP_T2_r3) + abs(meanP_T3_r3) + abs(meanP_T4_r3) + abs(meanP_T1_r4) + abs(meanP_T2_r4) + abs(meanP_T3_r4) + abs(meanP_T4_r4);
	fileNET << "Total rank probability bias:  " << to_string(TMB) << "\n";

	//Total Bias on SUCRA
	VectorXd data_R1(n_t);
	data_R1 << meanP_T1_r1, meanP_T2_r1, meanP_T3_r1, meanP_T4_r1;
	VectorXd data_R2(n_t);
	data_R2 << meanP_T1_r2, meanP_T2_r2, meanP_T3_r2, meanP_T4_r2;
	VectorXd data_R3(n_t);
	data_R3 << meanP_T1_r3, meanP_T2_r3, meanP_T3_r3, meanP_T4_r3;
	VectorXd data_R4(n_t);
	data_R4 << meanP_T1_r4, meanP_T2_r4, meanP_T3_r4, meanP_T4_r4;

	VectorXd BiasExp(n_t);		//Bias on expected rank
	VectorXd BiasSUCRA(n_t);	//Bias on SUCRA
	double sum_suc = 0.0;
	for(int i=0; i<n_t; i++){
		BiasExp(i) = data_R1(i)+2.0*data_R2(i)+3.0*data_R3(i)+4.0*data_R4(i);
		BiasSUCRA(i) = BiasExp(i)/(1.0-nt_d);
		sum_suc = sum_suc + abs(BiasSUCRA(i));
		file_SUCRA << to_string(BiasSUCRA(i)) << "\n";
	}
	fileNET<< "Total bias on SUCRA: " << to_string(sum_suc) << "\n";	

	//calculate mean of NT (\Bar{k})
	double sum = 0.0;
	for(int i=0; i<n_t; i++){
		sum = sum + NT(i);	
	}
	double k_bar = sum/nt_d;

	//calculate network irregularity h^2 = 1/nt sum_a (k_a - \Bar{k})^2
	double sum_diff = 0.0;
	for(int i=0; i<n_t; i++){
		sum_diff = sum_diff + pow((NT(i)-k_bar),2.0);	
	}
	double h = sum_diff/nt_d;

	//Normalised network irregularity h^2/k^2
	double h_norm = h/(pow(k_bar,2.0));

	fileNET << "h^2:  " << to_string(h) << "\n";
	fileNET << "k_bar:  " << to_string(k_bar) << "\n";
	fileNET << "h^2/kbar^2:  " << to_string(h_norm); 
	//-----------------------------------------------------------------------------------------
				

filesum.close();
file_avbias.close();
file_avSD.close();
fileP.close();
fileN.close();
fileNT.close();
fileNET.close();
file_SUCRA.close();

	
}//END OF INT MAIN

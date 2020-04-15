# Supplementary-Material-Davies-Galla-2020
Supplementary material and simulation code for 'Degree irregularity and rank probability bias in network meta-analysis'. A L Davies and T Galla 2020 https://arxiv.org/abs/2003.07662

The code simulates binomial data for a network meta-analysis with a combination of two-arm and multi-arm trials.
A Bayesian NMA is performed for each realisation of the data and the results are averaged.
The Bayesian NMA uses the Metropolis-in-Gibbs algorithm described in D. Spiegelhalter, A. Thomas, N. Best, and D. Lunn. WinBUGS User Manual: Version 1.4.MRC Biostatistics Unit, University of Cambridge, 2003.

The user must define the geometry of the network and the true model parameter values at the beginning of int main () {  }.
The code is written for a network of 4 treatments which are equally effective - changes to the code are required for a network of more/fewer treatments and/or non-equally effective treatments - these changes are commented where possible (NB: For different number of treatments, the user must change the number of model parameter vectors which store values at each iteration - we do not comment on this throughout the code). The user must check convergence of the MCMC for different set ups (e.g. using Brooks-Gelman-Rubin convergence diagnostics, see S. Brooks and A. Gelman. General methods for monitoring convergence of iterative simulations. Journal of Computational and Graphical Statistics, 7(4):434–455, 1998) and change parameters accordingly (i.e. the burn in, the number of Gibbs iterations and the proposal standard deviations). 

The code relies on the Eigen library, the download files and documentation can be found here: http://eigen.tuxfamily.org/index.php?title=Main_Page

The generation of random numbers (outside of functions) uses the PCG library, the download files and documentation can be found here: https://www.pcg-random.org/

Code written by Annabel Davies 09/2019
annabel.davies@postgrad.manchester.ac.uk

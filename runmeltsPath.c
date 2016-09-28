// runmeltsPlutonic.c: call alphaMELTS n times on n processors
// Last modified 1/25/15 by C. Brenhin Keller

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <mpi.h>
#include "arrays.h"
#include "runmelts.h"

int main(int argc, char **argv){
	MPI_Init(&argc, &argv);
	// Check input arguments
	if (argc != 3) {
		printf("USAGE: %s <sims_per_task> <residual_cutoff>\n", argv[0]);
		exit(1);
	}
	// Get number of simulations per MPI task from command-line argument
	const int sims_per_task = atoi(argv[1]);
	const double residual_cutoff = atof(argv[2]);

	// Starting composition
	//SiO2, TiO2, Al2O3, Fe2O3, Cr2O3, FeO, MnO, MgO, NiO, CoO, CaO, Na2O, K2O, P2O5, CO2, H2O
//	const double sc[16]={51.33, 0.98, 15.70, 0, 0.0582, 8.72, 0.17, 9.48, 0.0202, 0.0052, 9.93, 2.61, 0.88, 0.22, 1.14, 2.48}; // Primitive continental arc basalt, Kelemen 2014 TOG
//	const double sc[16]={48.978455,1.628352,15.461866,0.000000,0.033718,10.702109,0.182187,6.846443,0.014008,0.005441,9.518356,2.923306,1.114039,0.328446,0.499596,1.669180}; // Phanerozoic
	const double sc[16]={49.037538,0.895577,13.067472,0.000000,0.116847,11.453686,0.217967,10.298227,0.039579,0.007972,10.000358,1.798638,0.495633,0.108240,0.620580,2.101939}; // Archean

	double *ic=malloc(16*sizeof(double)), *icq=malloc(16*sizeof(double));

	// Composition path to fit/minimize to
	// SiO2, TiO2, Al2O3, Cr2O3, FeOT, MnO, MgO, Ni, Co, CaO, Na2O, K2O, P2O5
	const double composition[15][13] = 
/*		{{50.971056,1.437769,15.569346,0.029729,10.229630,0.181109,6.172354,0.010797,0.004871,8.783882,3.064027,1.272129,0.324383}
		{52.932414,1.281532,15.677081,0.024850,9.398641,0.172799,5.454910,0.008727,0.004450,8.066046,3.166448,1.609203,0.330014}
		{54.969145,1.132201,16.351373,0.016865,8.349070,0.184393,4.235200,0.006585,0.003304,6.587448,3.621741,2.201801,0.362470}
		{56.993829,0.992406,16.466098,0.011887,6.852526,0.180083,3.360997,0.004868,0.002646,5.470707,3.759509,2.896493,0.334940}
		{58.993268,0.858820,16.651507,0.010174,6.132129,0.131581,2.786968,0.004398,0.002225,4.697183,3.944367,3.107288,0.304112}
		{61.001616,0.773812,16.508710,0.007747,5.356855,0.130377,2.344216,0.003426,0.001910,4.252849,3.975700,3.039179,0.260930}
		{62.989888,0.699195,16.190852,0.005816,5.143399,0.120307,1.959255,0.002802,0.001631,3.820989,3.964665,3.041054,0.230836}
		{65.027051,0.606150,15.805755,0.004652,4.684041,0.110458,1.608531,0.002255,0.001334,3.225670,3.859647,3.251518,0.205634}
		{67.028119,0.511022,15.347414,0.003848,3.813866,0.092389,1.250660,0.001869,0.001147,2.707423,3.648396,3.576629,0.173695}
		{68.999655,0.420052,14.953063,0.003089,3.202183,0.086559,0.953872,0.001367,0.000905,2.159968,3.627341,3.732025,0.144194}
		{71.019714,0.340345,14.351912,0.002228,2.756470,0.082929,0.690190,0.001107,0.000743,1.667605,3.546693,4.006503,0.117597}
		{73.001034,0.249375,13.664794,0.001829,2.087063,0.071901,0.470360,0.000968,0.000707,1.197926,3.406177,4.296801,0.095299}
		{74.954576,0.194159,13.015031,0.001682,1.492367,0.085161,0.353426,0.000982,0.000703,0.809372,3.259008,4.385628,0.077585}
		{76.845341,0.184691,12.173429,0.001691,1.385537,0.066612,0.329988,0.000948,0.000744,0.595860,3.226511,4.089806,0.062458}
		{78.796039,0.237824,10.927241,0.002824,1.335570,0.047046,0.435741,0.001267,0.001105,0.513654,2.875245,3.396878,0.074728}};*/
	//PHANEROZOIC, raw means

	const double composition[15][13] = 
		{{50.976580,0.908733,13.229082,0.091078,11.207129,0.238004,9.002788,0.028967,0.007568,9.473752,2.124332,0.500550,0.117889}
		{52.924134,0.859020,13.252156,0.082241,10.344931,0.279310,8.471690,0.024583,0.007169,8.552533,2.362713,0.625641,0.130646}
		{54.945714,0.879617,13.927519,0.058305,9.598612,0.253407,6.760225,0.019249,0.006279,7.821518,2.759636,0.996128,0.173316}
		{56.910985,0.913264,14.429787,0.042164,8.790392,0.204236,5.029778,0.015705,0.005861,7.148607,3.193821,1.330573,0.202070}
		{58.970884,0.873982,15.033468,0.026986,7.760113,0.159800,4.105816,0.010818,0.004891,5.958605,3.338077,1.780239,0.240407}
		{61.012590,0.773667,15.354322,0.020057,6.683468,0.118727,3.237341,0.007810,0.003888,4.750258,3.799097,1.953141,0.221697}
		{63.048525,0.632740,15.636140,0.013451,5.588313,0.110935,2.539458,0.005863,0.003078,4.240711,3.998959,2.194318,0.206389}
		{65.018310,0.560457,15.829144,0.009627,4.522637,0.096114,2.026700,0.003866,0.002838,3.683781,4.255739,2.204342,0.187066}
		{67.024686,0.472415,15.625374,0.006306,3.544044,0.081822,1.531238,0.002765,0.002943,3.281635,4.390436,2.216987,0.156863}
		{69.043831,0.395157,15.319158,0.004773,2.890363,0.069419,1.145191,0.002740,0.003098,2.786249,4.424838,2.266331,0.122637}
		{71.011422,0.318271,14.941257,0.006112,2.297976,0.062807,0.807329,0.005081,0.003476,2.299074,4.558336,2.462456,0.098659}
		{72.950602,0.254452,14.176712,0.006724,1.983288,0.058554,0.611191,0.004078,0.003176,1.780971,4.126657,3.052076,0.076123}
		{74.912853,0.207849,13.316673,0.003907,1.820233,0.052448,0.507502,0.002117,0.002287,1.366115,3.663555,3.389888,0.058752}
		{76.881229,0.203397,12.366363,0.005409,1.619447,0.040361,0.429110,0.001300,0.001387,0.903590,3.228211,3.551985,0.058348}
		{78.817661,0.215150,11.601193,0.007867,1.640918,0.035456,0.586882,0.001417,0.001335,0.863312,2.691400,2.830786,0.047668}};
	//ARCHEAN, raw means



	// Simulation variables
	char prefix[200], cmd_string[500];
	FILE *fp;
	int minerals=0, i, j, k;

	// Get world size (number of MPI processes) and world rank (# of this process)
	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD,&world_size);
	MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
	printf("Hello from %d of %d processors\n", world_rank, world_size);

	// Variables that control size and location of the simulation
	/***********************************************************/	
	// Number of simulations to run , and temperature step size in each simulation
	const int nsims=world_size*sims_per_task, deltaT=-10;
	int n=world_rank-world_size;
	// Location of scratch directory (ideally local scratch for each node)
	// This location may vary on your system - contact your sysadmin if unsure
	const char scratchdir[]="/scratch/";
	/***********************************************************/	
	const int maxMinerals=40, maxSteps=1700/abs(deltaT), maxColumns=50;

	// Variables for MPI communication and reducing results
	const int sendArraySize=ceil(nsims/world_size);
	int *sendarrayN=calloc(sendArraySize,sizeof(int)), *rbufN, sendArrayIndex=0; 
	double *sendarrayR=calloc(sendArraySize,sizeof(double)), *rbufR, *sendarrayCO2=calloc(sendArraySize,sizeof(double)), *rbufCO2, *sendarrayWater=calloc(sendArraySize,sizeof(double)), *rbufWater, *sendarrayfo2=calloc(sendArraySize,sizeof(double)), *rbuffo2;

	
	// Malloc space for the imported array
	double **rawMatrix=mallocDoubleArray(maxMinerals*maxSteps,maxColumns);
	double ***melts=malloc(maxMinerals*sizeof(double**));
	char **names=malloc(maxMinerals*sizeof(char*));
	char ***elements=malloc(maxMinerals*sizeof(char**));
	int *rows=malloc(maxMinerals*sizeof(int)), *columns=malloc(maxMinerals*sizeof(int));
	for (i=0; i<maxMinerals; i++){
			names[i]=malloc(30*sizeof(char));
			elements[i]=malloc(maxColumns*sizeof(char*));
			for (k=0; k<maxColumns; k++){
				elements[i][k]=malloc(30*sizeof(char));
			}
	}

	// Variables for RNG and P-T path generator	
	const int steps=7;
	unsigned seed=(time(NULL)*world_rank) % 2147483000;
	double r1, r2, n1, n2, Pi, Pf, liquidusTemp, absc[steps], ordn[steps], x[500], y[500], fo2Delta;
	int points;

	//  Variables for minimization
	int SiO2, TiO2, Al2O3, Fe2O3, Cr2O3, FeO, MnO, MgO, NiO, CoO, CaO, Na2O, K2O, P2O5, CO2, H2O;
	double *residuals=malloc(maxSteps * sizeof(double)), *normconst=malloc(maxSteps * sizeof(double)), minresiduals=0;
	int meltsrow;

	if ( world_rank == 1) {	
	//	printf("   rank\t       n\t residual\t     CO2      \t     H2O\t     fO2\t      initial pressure\n"); // Command line output format
		printf("rank\tn\tresidual\tCO2\tH2O\tfO2\tinitial pressure\n"); // Command line output format
	}

	// Run the simulation
	while (n<nsims-world_size) {
		// Increment n
		n=n+world_size;

		// Set initial oxide composition
		for(i=0; i<14; i++){ic[i]=sc[i];}

		// Determine initial water and CO2 content
		ic[14]=rand_r(&seed)/(double)RAND_MAX*1;
		ic[15]=rand_r(&seed)/(double)RAND_MAX*4;

		// Determine inital fO2
		fo2Delta=0;

		// Determine starting pressure
		r1=rand_r(&seed)/(double)RAND_MAX, r2=rand_r(&seed)/(double)RAND_MAX; //generate two standard uniform variables
		Pi=10000+15000*r1, Pf=10000*r2;

		// Configure working directory
		sprintf(prefix,"%sout%d/", scratchdir, n);
		sprintf(cmd_string,"mkdir -p %s", prefix);
		system(cmd_string);

		// Run MELTS to equilibrate fO2 at the liquidus
		runmelts(prefix,ic,"pMELTS","isobaric","FMQ",fo2Delta,"1\nsc.melts\n10\n1\n3\n1\nliquid\n1\n0.99\n0\n0\n","","!",1700,Pi,deltaT,0,0.005);
		
		// If simulation failed, clean up scratch directory and move on to next simulation
		sprintf(cmd_string,"%sPhase_main_tbl.txt", prefix);
		if ((fp = fopen(cmd_string, "r")) == NULL) {
			fprintf(stderr, "%d: MELTS equilibration failed to produce output.\n", n);
			sprintf(cmd_string,"rm -r %s", prefix);
			system(cmd_string);
			continue;
		}
		importmelts(prefix, melts, rawMatrix, rows, columns, names, elements, &minerals); // Import results, if they exist
		if (minerals<1 | strcmp(names[0],"liquid_0")!=0) {
			fprintf(stderr, "%d: MELTS equilibration failed to calculate liquid composition.\n", n);
			sprintf(cmd_string,"rm -r %s", prefix);
			system(cmd_string);
			continue;
		}

		// Copy the liquid composition for use as the starting composition of the next MELTS calculation. Format:
		// Pressure Temperature mass S H V Cp viscosity SiO2 TiO2 Al2O3 Fe2O3 Cr2O3 FeO MnO MgO NiO CoO CaO Na2O K2O P2O5 CO2 H2O
		for (i=0; i<16; i++){
			icq[i]=melts[0][0][i+8];
 		}
		liquidusTemp=melts[0][0][1];
		points=(int)ceil((liquidusTemp-700)/(-deltaT));


		// Generate x and y vectors for P-T path
		absc[0]=0;
		ordn[0]=NAN;
		while isnan(ordn[0]){
			ordn[0]=0;
			for (i=1; i<steps; i++){
				if ((double)rand_r(&seed)/(double)RAND_MAX > 0.5){
					ordn[i]=ordn[i-1]-(double)rand_r(&seed)/(double)RAND_MAX;
					absc[i]=absc[i-1]+(double)rand_r(&seed)/(double)RAND_MAX/2;
				} else {
					ordn[i]=ordn[i-1];
					absc[i]=absc[i-1]+(double)rand_r(&seed)/(double)RAND_MAX;
				}
			}
			for(i=0; i<steps; i++) {
				ordn[i]=ordn[i]-ordn[steps-1]; //Align y axis
				absc[i]=absc[i] * points / absc[steps-1]; //Scale x axis
			}
			for(i=steps-1; i>=0; i--) ordn[i]=ordn[i] * (Pi-Pf) / ordn[0] + Pf; //Scale y axis
		}

		for (i=1; i<steps; i++){
			for (j=floor(absc[i-1]); j<floor(absc[i]) & j<500; j++) {
				y[j]=ordn[i-1] + (j+1-absc[i-1]) / (absc[i]-absc[i-1]) * (ordn[i]-ordn[i-1]);
				x[j]=j+1;
			}
		}

		// Write P-T path to file
		sprintf(cmd_string,"%sptpath", prefix);
		fp=fopen(cmd_string, "w");
		for (i=0; i<points; i++) fprintf(fp,"%f %f\n",y[i],liquidusTemp+deltaT*i);
		fclose(fp);
	
		minerals=0;
	
		// Run MELTS
		runmelts(prefix,icq,"pMELTS","PTpath","None",0.0,"1\nsc.melts\n10\n1\n3\n1\nliquid\n1\n0.99\n0\n10\n0\n4\n0\n","!","",1700,Pi,deltaT,0,0.005);
	
		// If simulation failed, clean up scratch directory and move on to next simulation
		if ((fp = fopen(cmd_string, "r")) == NULL) {
			fprintf(stderr, "%d: melts simulation failed to produce output.\n", n);
			sprintf(cmd_string,"rm -r %s", prefix);
			system(cmd_string);
			continue;
		}
		importmelts(prefix, melts, rawMatrix, rows, columns, names, elements, &minerals); // Import results, if they exist
		if (minerals<1 | strcmp(names[0],"liquid_0")!=0) {
			fprintf(stderr, "%d: MELTS simulation failed to calculate liquid composition.\n", n);
			sprintf(cmd_string,"rm -r %s", prefix);
			system(cmd_string);
			continue;
		}

		// Find the columns containing useful elements
		for(i=0; i<columns[0]; i++){
			if (strcmp(elements[0][i], "SiO2")==0) SiO2=i;
			else if (strcmp(elements[0][i], "TiO2")==0) TiO2=i;
			else if (strcmp(elements[0][i], "Al2O3")==0) Al2O3=i;
			else if (strcmp(elements[0][i], "Fe2O3")==0) Fe2O3=i;
			else if (strcmp(elements[0][i], "Cr2O3")==0) Cr2O3=i;
			else if (strcmp(elements[0][i], "FeO")==0) FeO=i;
			else if (strcmp(elements[0][i], "MnO")==0) MnO=i;
			else if (strcmp(elements[0][i], "MgO")==0) MgO=i;
			else if (strcmp(elements[0][i], "NiO")==0) NiO=i;
			else if (strcmp(elements[0][i], "CoO")==0) CoO=i;
			else if (strcmp(elements[0][i], "CaO")==0) CaO=i;
			else if (strcmp(elements[0][i], "Na2O")==0) Na2O=i;
			else if (strcmp(elements[0][i], "K2O")==0) K2O=i;
			else if (strcmp(elements[0][i], "P2O5")==0) P2O5=i;
			else if (strcmp(elements[0][i], "CO2")==0) CO2=i;
			else if (strcmp(elements[0][i], "H2O")==0) H2O=i;
		}



		// Renormalize melts output
		for (meltsrow=0; meltsrow<rows[0]; meltsrow++){
			normconst[meltsrow]=1; //No normalization
//			normconst[meltsrow]=100/(100-melts[0][meltsrow][CO2]-melts[0][meltsrow][H2O]); //Anhydrous normalization
		}

		// Find how well this melts simulation matches the composition we're fitting to
		minresiduals=0;
		for (i=0; i<15; i++){
			for (meltsrow=0; meltsrow<rows[0]; meltsrow++){
				residuals[meltsrow]=(pow((composition[i][0] - normconst[meltsrow]*melts[0][meltsrow][SiO2]),2)*2 + 
						pow((composition[i][1] - normconst[meltsrow]*melts[0][meltsrow][TiO2]),2) + 
						pow((composition[i][2] - normconst[meltsrow]*melts[0][meltsrow][Al2O3]),2) + 
						pow((composition[i][3] - normconst[meltsrow]*melts[0][meltsrow][Cr2O3]),2) + 
						pow((composition[i][4] - normconst[meltsrow]*melts[0][meltsrow][FeO]-melts[0][meltsrow][Fe2O3]/1.1113),2) + 
						pow((composition[i][5] - normconst[meltsrow]*melts[0][meltsrow][MnO]),2) + 
						pow((composition[i][6] - normconst[meltsrow]*melts[0][meltsrow][MgO]),2) + 
						pow((composition[i][7] - normconst[meltsrow]*melts[0][meltsrow][NiO]),2) + 
						pow((composition[i][8] - normconst[meltsrow]*melts[0][meltsrow][CoO]),2) + 
						pow((composition[i][9] - normconst[meltsrow]*melts[0][meltsrow][CaO]),2) + 
						pow((composition[i][10] - normconst[meltsrow]*melts[0][meltsrow][Na2O]),2) + 
						pow((composition[i][11] - normconst[meltsrow]*melts[0][meltsrow][K2O]),2)*2 + 
						pow((composition[i][12] - normconst[meltsrow]*melts[0][meltsrow][P2O5]),2));
			}
			minresiduals=minresiduals+minArray(residuals, rows[0]);
		}
		
		// Print results to command line
//		printf("%7d\t%8d\t%14.8f\t%15.8f\t%15.8f\t%15.8f\t%20.8f\n", world_rank, n, minresiduals, ic[14], ic[15], fo2Delta, Pi); % with pretty formatting
		printf("%d\t%d\t%f\t%f\t%f\t%f\t%f\n", world_rank, n, minresiduals, ic[14], ic[15], fo2Delta, Pi);

		sendarrayN[sendArrayIndex]=n;
		sendarrayR[sendArrayIndex]=minresiduals;
		sendarrayCO2[sendArrayIndex]=ic[14];
		sendarrayWater[sendArrayIndex]=ic[15];
		sendarrayfo2[sendArrayIndex]=fo2Delta;

		
		// Copy useful output to current directory
		if  (minresiduals < residual_cutoff){ // Recommended general: 200
			sprintf(cmd_string,"cp -r %s ./out%d/", prefix, n);
			system(cmd_string);
		}
		// Clean up scratch directory
		sprintf(cmd_string,"rm -r %s", prefix);
		system(cmd_string);
		

		minerals=0;
		sendArrayIndex++;
	}


	if ( world_rank == 0) {
		rbufN = (int *)malloc(world_size*sendArraySize*sizeof(int));  
		rbufR = (double *)malloc(world_size*sendArraySize*sizeof(double));
		rbufCO2 = (double *)malloc(world_size*sendArraySize*sizeof(double));
		rbufWater = (double *)malloc(world_size*sendArraySize*sizeof(double));
		rbuffo2 = (double *)malloc(world_size*sendArraySize*sizeof(double));

	} 
	MPI_Gather(sendarrayN, sendArraySize, MPI_INT, rbufN, sendArraySize, MPI_INT, 0, MPI_COMM_WORLD); 
	MPI_Gather(sendarrayR, sendArraySize, MPI_DOUBLE, rbufR, sendArraySize, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
	MPI_Gather(sendarrayCO2, sendArraySize, MPI_DOUBLE, rbufCO2, sendArraySize, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
	MPI_Gather(sendarrayWater, sendArraySize, MPI_DOUBLE, rbufWater, sendArraySize, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
	MPI_Gather(sendarrayfo2, sendArraySize, MPI_DOUBLE, rbuffo2, sendArraySize, MPI_DOUBLE, 0, MPI_COMM_WORLD); 

	// Save results to file
	if (world_rank==0){
		fp=fopen("residuals.csv","w");
		fprintf(fp,"n\tresidual\tCO2\tH2O\tfO2\n"); // File output format
		for (n=0; n<nsims; n++){
			fprintf(fp,"%d\t%f\t%f\t%f\t%f\n", rbufN[n], rbufR[n], rbufCO2[n], rbufWater[n], rbuffo2[n]);
		}
		fclose(fp);
	}

	free(residuals);
	free(normconst);

	MPI_Finalize();
	return 0;
}

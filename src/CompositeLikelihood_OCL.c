#include "header.h"
/******************************************************************************************/
/******************************************************************************************/
/******************************************************************************************/
/********************* SPATIAL CASE *****************************************************/
/******************************************************************************************/
/******************************************************************************************/
/******************************************************************************************/
/******************************************************************************************/
/******************************************************************************************/
/******************************************************************************************/
// Composite conditional log-likelihood for the spatial Gaussian model:
void Comp_Cond_Gauss2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
                    double *par,  int *weigthed,double *res,double *mean,double *mean2,double *nuis,int *ns, int *NS,
                    int *local_wi, int *dev)
{
    char *f_name = "Comp_Cond_Gauss2_OCL";

    int *int_par;
    double *dou_par;
    int_par = (int*)calloc((50), sizeof(int));
    dou_par = (double*)calloc((50), sizeof(double));
    param_OCL(cormod,NN,par,weigthed,nuis,int_par,dou_par);
    //exec_kernel(coordx,coordy,mean, data, int_par, dou_par, local_wi,dev,res,f_name);
	exec_kernel_source(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
    
    
    if(!R_FINITE(*res))*res = LOW;  
}

// Composite marginal (pariwise) log-likelihood for the spatial Gaussian model:
void Comp_Pair_Gauss2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS,
	int *local_wi, int *dev)
{

	double sill, nugget;
	sill = nuis[1]; nugget = nuis[0];
	if (sill<0 || nugget<0) { *res = LOW;  return; }

	char *f_name = "Comp_Pair_Gauss2_OCL";
	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	//exec_kernel(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	exec_kernel_source(coordx,coordy,mean, data, int_par, dou_par, local_wi,dev,res,f_name);
	if (!R_FINITE(*res))*res = LOW;
}

// Composite marginal (difference) log-likelihood for the spatial Gaussian model:
void Comp_Diff_Gauss2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	//if(nuis[1]<0 || nuis[0]<0) {*res=LOW;  return;}
	char *f_name = "Comp_Diff_Gauss2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);
	//exec_kernel(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	exec_kernel_source(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	if (!R_FINITE(*res))*res = LOW;
}



void Comp_Pair_WrapGauss2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	//if(nuis[1]<0 || nuis[0]<0) {*res=LOW;  return;}
	char *f_name = "Comp_Pair_WrapGauss2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);
	//exec_kernel(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	exec_kernel_source(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	if (!R_FINITE(*res))*res = LOW;

}

void Comp_Pair_SinhGauss2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	//if(nuis[1]<0 || nuis[0]<0|| nuis[3]<0) {*res=LOW;  return;}
	char *f_name = "Comp_Pair_SinhGauss2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);
	exec_kernel_source(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	//exec_kernel(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	if (!R_FINITE(*res))*res = LOW;
}



void Comp_Pair_LogGauss2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	if (nuis[1]<0 || nuis[0]<0) { *res = LOW;  return; }
	char *f_name = "Comp_Pair_LogGauss2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);
	exec_kernel_source(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	//exec_kernel(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	if (!R_FINITE(*res))*res = LOW;
}




void Comp_Pair_Gamma2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	double sill = 1 - nuis[0];
	if (nuis[2]<1 || sill<0 || sill>1) { *res = LOW;  return; }
	char *f_name = "Comp_Pair_Gamma2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);
	exec_kernel_source(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	//exec_kernel(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	//printf("resantes:::%f:::\n",*res);
	if (!R_FINITE(*res)) *res = LOW;
}


void Comp_Pair_BinomGauss2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	if (nuis[0]>1 || nuis[0]<0) { *res = LOW; return; }

	char *f_name = "Comp_Pair_BinomGauss2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	//exec_kernel(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	exec_kernel_source(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	if (!R_FINITE(*res))*res = LOW;
}


void Comp_Pair_BinomnegGauss2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	if (nuis[0]>1 || nuis[0]<0) { *res = LOW; return; }

	char *f_name = "Comp_Pair_BinomnegGauss2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	//exec_kernel(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	exec_kernel_source(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	if (!R_FINITE(*res))*res = LOW;
}

void Comp_Pair_SkewGauss2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	//if(nuis[1]<0 || nuis[0]<0) {*res=LOW;  return;}
	char *f_name = "Comp_Pair_SkewGauss2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);
	exec_kernel_source(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	//exec_kernel(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	if (!R_FINITE(*res))*res = LOW;
}

void Comp_Pair_PoisbinGauss2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	if (nuis[0]>1 || nuis[0]<0) { *res = LOW; return; }

	char *f_name = "Comp_Pair_PoisbinGauss2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	//exec_kernel(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	exec_kernel_source(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	if (!R_FINITE(*res))*res = LOW;
}


void Comp_Pair_PoisbinnegGauss2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	if (nuis[0]>1 || nuis[0]<0) { *res = LOW; return; }

	char *f_name = "Comp_Pair_PoisbinnegGauss2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);
	exec_kernel_source(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	//exec_kernel(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	if (!R_FINITE(*res))*res = LOW;
}


void Comp_Pair_Logistic2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data,
	int *NN, double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{

	if (nuis[1] <= 0) { *res = LOW;  return; }
	char *f_name = "Comp_Pair_Logistic2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

//	exec_kernel(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	exec_kernel_source(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);

	if (!R_FINITE(*res)) *res = LOW;
}



void Comp_Pair_LogLogistic2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data,
	int *NN, double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{


	char *f_name = "Comp_Pair_LogLogistic2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	//exec_kernel(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	exec_kernel_source(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);

	if (!R_FINITE(*res)) *res = LOW;
}

void Comp_Pair_Weibull2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data,
	int *NN, double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{


	double sill = 1 - nuis[0];
	if (nuis[2] <= 0 || sill<0 || sill>1) { *res = LOW;  return; }
	char *f_name = "Comp_Pair_Weibull2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	//exec_kernel(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	exec_kernel_source(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);

	if (!R_FINITE(*res)) *res = LOW;
}


// Composite marginal (pariwise) log-likelihood for the spatial Gaussian model:
void Comp_Pair_T2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS,
	int *local_wi, int *dev)
{
	double sill = nuis[2];
	double nugget = nuis[1];
	double df = nuis[0];
	//if (sill<0 || df<0 || df>0.5) { *res = LOW; return; }
	if (sill<0 || df<0 || df>0.5 || nugget >= 1 || nugget<0) { *res = LOW; return; }

	char *f_name = "Comp_Pair_T2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	//exec_kernel(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	exec_kernel_source(coordx,coordy,mean, data, int_par, dou_par, local_wi,dev,res,f_name);
	if (!R_FINITE(*res))*res = LOW;
}


void Comp_Pair_TWOPIECET2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS,
	int *local_wi, int *dev)
{
	double eta = nuis[3];  //skewness parameter
	double sill = nuis[2];
	double df = nuis[0];

	if (fabs(eta)>1 || sill<0 || df >0.5 || df<0) { *res = LOW;  return; }

	char *f_name = "Comp_Pair_TWOPIECET2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	//exec_kernel(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	exec_kernel_source(coordx,coordy,mean, data, int_par, dou_par, local_wi,dev,res,f_name);
	if (!R_FINITE(*res))*res = LOW;
}


void Comp_Pair_TWOPIECEGauss2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS,
	int *local_wi, int *dev)
{
	double eta = nuis[2];  //skewness parameter
	double sill = nuis[1];
	double nugget = nuis[0];

	//if (fabs(eta)>1 || sill<0) { *res = LOW;  return; }
	if (fabs(eta)>1 || sill<0 || nugget >= 1 || nugget<0) { *res = LOW;  return; }

	char *f_name = "Comp_Pair_TWOPIECEGauss2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	//exec_kernel(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
	exec_kernel_source(coordx,coordy,mean, data, int_par, dou_par, local_wi,dev,res,f_name);
	if (!R_FINITE(*res))*res = LOW;
}



/******************************************************************************************/
/******************************************************************************************/
/******************************************************************************************/
/********************* SPACE TIME CASE *****************************************************/
/******************************************************************************************/
/******************************************************************************************/
/******************************************************************************************/
/******************************************************************************************/
/******************************************************************************************/
/******************************************************************************************/
// Composite marginal (pariwise) log-likelihood for the spatial-temporal Gaussian model:

void Comp_Pair_Gauss_st2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	if (nuis[1]<0 || nuis[0]<0) { *res = LOW;  return; }
	char *f_name = "Comp_Pair_Gauss_st2_OCL";
	//printf("WEIGHTED: %d\n", *weigthed);
	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_st_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	if (cdyn[0] == 0)
	{
		//printf("*****exec_kernel_st_source.c\n");
		//exec_kernel_st(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		//exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	else
	{
		//printf("*****exec_kernel_st_dyn_source.c\n");
		//exec_kernel_st_dyn(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	if (!R_FINITE(*res))*res = LOW;
}

void Comp_Pair_WrapGauss_st2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	//if(nuis[1]<0 || nuis[0]<0) {*res=LOW;  return;}
	char *f_name = "Comp_Pair_WrapGauss_st2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_st_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	if (cdyn[0] == 0)
	{
		//exec_kernel_st(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	else
	{
		//exec_kernel_st_dyn(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	if (!R_FINITE(*res))*res = LOW;
}

void Comp_Pair_PoisbinGauss_st2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	if (nuis[0]>1 || nuis[0]<0) { *res = LOW; return; }
	char *f_name = "Comp_Pair_PoisbinGauss_st2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_st_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	if (cdyn[0] == 0)
	{
		//exec_kernel_st(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	else
	{
		//exec_kernel_st_dyn(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	if (!R_FINITE(*res))*res = LOW;
}


void Comp_Pair_PoisbinnegGauss_st2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	if (nuis[0]>1 || nuis[0]<0) { *res = LOW; return; }
	char *f_name = "Comp_Pair_PoisbinnegGauss_st2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_st_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	if (cdyn[0] == 0)
	{
		//exec_kernel_st(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	else
	{
		//exec_kernel_st_dyn(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	if (!R_FINITE(*res))*res = LOW;
}



void Comp_Cond_Gauss_st2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	//if(nuis[1]<0 || nuis[0]<0) {*res=LOW;  return;}
	char *f_name = "Comp_Cond_Gauss_st2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_st_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	if (cdyn[0] == 0)
	{
		//exec_kernel_st(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	else
	{
		//exec_kernel_st_dyn(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	if (!R_FINITE(*res))*res = LOW;
}


void Comp_Diff_Gauss_st2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	//if(nuis[1]<0 || nuis[0]<0) {*res=LOW;  return;}
	char *f_name = "Comp_Diff_Gauss_st2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_st_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	if (cdyn[0] == 0)
	{
		//exec_kernel_st(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	else
	{
		//exec_kernel_st_dyn(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	if (!R_FINITE(*res))*res = LOW;
}

void Comp_Pair_SkewGauss_st2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	//if(nuis[1]<0 || nuis[0]<0) {*res=LOW;  return;}
	char *f_name = "Comp_Pair_SkewGauss_st2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_st_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	if (cdyn[0] == 0)
	{
		//exec_kernel_st(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	else
	{
		//exec_kernel_st_dyn(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	if (!R_FINITE(*res))*res = LOW;
}

void Comp_Pair_SinhGauss_st2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	//if(nuis[1]<0 || nuis[0]<0|| nuis[3]<0) {*res=LOW;  return;}
	char *f_name = "Comp_Pair_SinhGauss_st2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_st_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	if (cdyn[0] == 0)
	{
		//exec_kernel_st(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	else
	{
		//exec_kernel_st_dyn(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	if (!R_FINITE(*res))*res = LOW;
}


void Comp_Pair_Gamma_st2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	double sill = 1 - nuis[0];
	if (nuis[2]<1 || sill<0 || sill>1) { *res = LOW;  return; }
	char *f_name = "Comp_Pair_Gamma_st2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_st_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);
	if (cdyn[0] == 0)
	{
		//exec_kernel_st(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	else
	{
		//exec_kernel_st_dyn(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}


	if (!R_FINITE(*res))*res = LOW;
}




void Comp_Pair_LogGauss_st2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	if (nuis[1]<0 || nuis[0]<0) { *res = LOW;  return; }
	char *f_name = "Comp_Pair_LogGauss_st2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_st_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	if (cdyn[0] == 0)
	{
		//exec_kernel_st(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	else
	{
		//exec_kernel_st_dyn(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	if (!R_FINITE(*res))*res = LOW;

}


void Comp_Pair_BinomGauss_st2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	if (nuis[0]>1 || nuis[0]<0) { *res = LOW; return; }
	char *f_name = "Comp_Pair_BinomGauss_st2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_st_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	if (cdyn[0] == 0)
	{
		//exec_kernel_st(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	else
	{
		//exec_kernel_st_dyn(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	if (!R_FINITE(*res))*res = LOW;

}



void Comp_Pair_BinomnegGauss_st2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	if (nuis[0]>1 || nuis[0]<0) { *res = LOW; return; }
	char *f_name = "Comp_Pair_BinomnegGauss_st2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_st_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	if (cdyn[0] == 0)
	{
		//exec_kernel_st(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	else
	{
		//exec_kernel_st_dyn(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	if (!R_FINITE(*res))*res = LOW;

}


void Comp_Pair_LogLogistic_st2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	//if( nuis[0]>1 || nuis[0]<0){*res=LOW; return;}
	char *f_name = "Comp_Pair_LogLogistic_st2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_st_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	if (cdyn[0] == 0)
	{
		//exec_kernel_st(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	else
	{
		//exec_kernel_st_dyn(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	if (!R_FINITE(*res))*res = LOW;

}

void Comp_Pair_Logistic_st2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	//if( nuis[2]<=0)  {*res=LOW;  return;}
	char *f_name = "Comp_Pair_Logistic_st2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_st_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	if (cdyn[0] == 0)
	{
		//exec_kernel_st(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	else
	{
		//exec_kernel_st_dyn(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	if (!R_FINITE(*res))*res = LOW;

}

void Comp_Pair_Weibull_st2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	double sill = 1 - nuis[0];
	if (nuis[2] <= 0 || sill<0 || sill>1) { *res = LOW;  return; }
	char *f_name = "Comp_Pair_Weibull_st2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_st_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	if (cdyn[0] == 0)
	{
		//exec_kernel_st(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	else
	{
		//exec_kernel_st_dyn(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	if (!R_FINITE(*res))*res = LOW;
}

void Comp_Pair_T_st2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	double sill = nuis[2];
	double df = nuis[0];
	if (sill<0 || df<0 || df>0.5) { *res = LOW; return; }
	char *f_name = "Comp_Pair_T_st2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_st_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	if (cdyn[0] == 0)
	{
		//exec_kernel_st(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	else
	{
		//exec_kernel_st_dyn(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}

	if (!R_FINITE(*res))*res = LOW;
}


void Comp_Pair_TWOPIECET_st2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
	double *par, int *weigthed, double *res, double *mean, double *mean2, double *nuis, int *ns, int *NS, int *local_wi, int *dev)
{
	double eta = nuis[3];  //skewness parameter
	double sill = nuis[2];
	double df = nuis[0];
	if (fabs(eta)>1 || sill<0 || df >0.5 || df<0) { *res = LOW;  return; }
	char *f_name = "Comp_Pair_TWOPIECET_st2_OCL";

	int *int_par;
	double *dou_par;
	int_par = (int*)calloc((50), sizeof(int));
	dou_par = (double*)calloc((50), sizeof(double));
	param_st_OCL(cormod, NN, par, weigthed, nuis, int_par, dou_par);

	if (cdyn[0] == 0)
	{
		//exec_kernel_st(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}
	else
	{
		//exec_kernel_st_dyn(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
		exec_kernel_st_dyn_source(coordx, coordy, coordt, mean, data, int_par, dou_par, local_wi, dev, res, f_name, ns, NS);
	}

	if (!R_FINITE(*res))*res = LOW;
}
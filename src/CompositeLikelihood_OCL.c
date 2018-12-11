#include "header.h"
// Composite conditional log-likelihood for the spatial Gaussian model:
void Comp_Cond_Gauss2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
                    double *par,  int *weigthed,double *res,double *mean,double *mean2,double *nuis,int *ns, int *NS,
                    int *local_wi, int *dev)
{

	//printf("A. ME DUERMOOOOOO\n");
    char *f_name = "Comp_Cond_Gauss2_OCL";

    int *int_par;
    double *dou_par;
    int_par = (int*)calloc((50), sizeof(int));
    dou_par = (double*)calloc((50), sizeof(double));
    param_OCL(cormod,NN,par,weigthed,nuis,int_par,dou_par);
    //printf("%f\t%f\n",dou_par[6],maxdist[0]);
    //exec_kernel(coordx,coordy,mean, data, int_par, dou_par, local_wi,dev,res,f_name);
	exec_kernel_source(coordx, coordy, mean, data, int_par, dou_par, local_wi, dev, res, f_name);
    
    
    if(!R_FINITE(*res))*res = LOW;
    //free(int_par);
    //free(dou_par);
    
    
}

// Composite marginal (pariwise) log-likelihood for the spatial Gaussian model:
void Comp_Pair_Gauss2_OCL(int *cormod, double *coordx, double *coordy, double *coordt, double *data, int *NN,
                          double *par,  int *weigthed,double *res,double *mean,double *mean2,double *nuis,int *ns, int *NS,
                          int *local_wi, int *dev)
{
  
    double sill,nugget;
    sill=nuis[1];nugget=nuis[0];
    if(sill<0||nugget<0) {*res=LOW;  return;}
    
    char *f_name = "Comp_Pair_Gauss2_OCL";
    int *int_par;
    double *dou_par;
    int_par = (int*)calloc((50), sizeof(int));
    dou_par = (double*)calloc((50), sizeof(double));
    param_OCL(cormod,NN,par,weigthed,nuis,int_par,dou_par);
    
    exec_kernel(coordx,coordy, mean,data, int_par, dou_par, local_wi,dev,res,f_name);
    //exec_kernel_source(coordx,coordy,mean, data, int_par, dou_par, local_wi,dev,res,f_name);
    if(!R_FINITE(*res))*res = LOW;
}



// Composite marginal (difference) log-likelihood for the spatial Gaussian model:
void Comp_Diff_Gauss2_OCL(int *cormod, double *coordx, double *coordy, double *coordt,double *data, int *NN,
                      double *par, int *weigthed, double *res,double *mean,double *mean2,double *nuis,int *ns, int *NS,int *local_wi, int *dev)
{
    //if(nuis[1]<0 || nuis[0]<0) {*res=LOW;  return;}
    char *f_name = "Comp_Diff_Gauss2_OCL";
    
    int *int_par;
    double *dou_par;
    int_par = (int*)calloc((50), sizeof(int));
    dou_par = (double*)calloc((50), sizeof(double));
    param_OCL(cormod,NN,par,weigthed,nuis,int_par,dou_par);
    exec_kernel(coordx,coordy, mean,data, int_par, dou_par, local_wi,dev,res,f_name);

    if(!R_FINITE(*res))*res = LOW;
}

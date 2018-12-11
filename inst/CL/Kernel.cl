double Dist_chordal(double loni, double lati, double lonj, double latj,double radius);
double Dist_geodesic(double loni, double lati, double lonj, double latj,double radius);
double dist(int type_dist,double coordx,double locx,double coordy,double locy,double radius);
double CorFunCauchy(double lag, double R_power2, double scale);
double CorFunStable(double lag, double R_power, double scale);
double CorFct(int cormod, double h, double u, double par0,double par1,double par2,double par3, int c11, int c22);
double Variogram(int cormod, double h, double u, double nugget, double var, double par0,double par1,double par2,double par3);
double CorFunBohman(double lag,double scale);
double log_biv_Norm(double corr,double zi,double zj,double mi,double mj,double vari, double nugget);
// Utility.c
double Dist_chordal(double loni, double lati, double lonj, double latj,double radius)
{
    double ai, bi, aj, bj, val=0.0;
    if (loni == lonj && lati == latj) return val;
    ai = (lati)*M_PI/180;
    bi = (loni)*M_PI/180;
    aj = (latj)*M_PI/180;
    bj = (lonj)*M_PI/180;
    val=radius  *sqrt(pow(cos(ai) * cos(bi)-cos(aj)  *cos(bj) ,2) +
                      pow(cos(ai) * sin(bi)-cos(aj) * sin(bj) ,2)+
                      pow(sin(ai)-sin(aj) ,2));
    return(val);
}

// Computes the Geodesic distance between to coordinates:
double Dist_geodesic(double loni, double lati, double lonj, double latj,double radius)
{
    double ai, bi, aj, bj, val=0.0,val2=0.0;
    if (loni == lonj && lati == latj) return val;
    ai = (lati)*M_PI/180;
    bi = (loni)*M_PI/180;
    aj = (latj)*M_PI/180;
    bj = (lonj)*M_PI/180;
    val = sin(ai) * sin(aj) + cos(ai) * cos(aj) * cos(bi - bj);
    if(val<= -1)  val2=M_PI*radius;
    if(val>=1) val2=0;
    val2 = acos(val)*radius;
    return(val2);
}

double dist(int type_dist,double coordx,double locx,double coordy,double locy,double radius)
{
    double lags=0.0;
    
    if(type_dist==0) lags=hypot(coordx-locx,coordy-locy);                        //euclidean
    if(type_dist==2) lags=Dist_geodesic(coordx,coordy,locx,locy,radius);           //great circle
    if(type_dist==1) lags=Dist_chordal(coordx,coordy,locx,locy,radius);      //chordal
    
    return(lags);
}

// ===================================== END Distance Functions  =====================================//

// ===================================== START CorrelationFunction.c  ==================================//
// Cauhcy class of correlation models:
double CorFunCauchy(double lag, double R_power2, double scale)
{
    double rho=0.0;
    // Computes the correlation:
    rho=pow((1+pow(lag/scale,2)),-R_power2/2);
    return rho;
}

// Stable class of correlation models:
double CorFunStable(double lag, double R_power, double scale)
{
    double rho=0.0;
    // Computes the correlation:
    rho=exp(-pow(lag/scale,R_power));
    return rho;
}


double CorFct(int cormod, double h, double u, double par0,double par1,double par2,double par3, int c11, int c22)
{
    double arg=0.0, col=0.0,R_power=0.0, R_power1=0.0, R_power2=0.0, R_power_s=0.0, R_power_t=0.0, var11=0.0, var22=0.0;
    double rho=0.0, sep=0, scale=0.0, smooth=0.0,smooth_s=0.0,smooth_t=0.0, scale_s=0.0, scale_t=0, x=0, nug11=0.0, nug22=0.0;
    double scale11=0.0, scale22=0.0, scale12=0.0, smoo11=0.0, smoo22=0.0, smoo12=0.0,R_power11=0.0, R_power22=0.0, R_power12=0.0;
    switch(cormod) // Correlation functions are in alphabetical order
    {
            // ========================   SPACE
        case 1:// Cauchy correlation function
            R_power1=2;
            R_power2=par0;
            scale=par1;
            rho=CorFunCauchy(h, R_power2, scale);
            break;
        
        case 4:// Exponential correlation function
            R_power=1;
            scale=par0;
            rho=CorFunStable(h, R_power, scale);
            break;

    }
    return rho;
}



// Computes the spatio-temporal variogram:
double Variogram(int cormod, double h, double u, double nugget, double var, double par0,double par1,double par2,double par3)
{
    double vario=0.0;
    //Computes the variogram
    vario=nugget+var*(1-CorFct(cormod,h,u,par0,par1,par2,par3,0,0));
    return vario;
}


double CorFunBohman(double lag,double scale)
{
    double rho=0.0,x=0;
    x=lag/scale;
    if(x<=1)
    {
        if (x>0) rho=(1-x)*(sin(2*M_PI*x)/(2*M_PI*x))+(1-cos(2*M_PI*x))/(2*M_PI*M_PI*x);
        else   rho=1;
    }
    else rho=0;
    return rho;
}


// ===================================== END CorrelationFunction.c  ==================================//




// END aux fun for biv_T

// ===================================== START: Distributions.c  ==================================//

double log_biv_Norm(double corr,double zi,double zj,double mi,double mj,double vari, double nugget)
{
    double u,v,u2,v2,det,s1,s12,dens;
    u=zi-mi;
    v=zj-mj;
    u2=pow(u,2);v2=pow(v,2);
    s1=vari+nugget;s12=vari*corr;
    det=pow(s1,2)-pow(s12,2);
    dens=(-0.5*(2*log(2*M_PI)+log(det)+(s1*(u2+v2)-2*s12*u*v)/det));
    return(dens);
}
// ===================================== END: Distributions.c  ==================================//

/******************************************************************************************/
/********************* SPATIAL CASE *****************************************************/
/******************************************************************************************/
__kernel void Comp_Cond_Gauss2_OCL(__global const double *coordx,__global const double *coordy,__global const double *mean, __global const double *data, __global double *res,__global const int *int_par,__global const double *dou_par)
{
    
    int j, gid = get_global_id(0);
    double s1=0.0, s12=0.0, lags=0.0,weights=1.0, sum=0.0;
    double det=0.0, u=0.0, u2=0.0, v=0.0, v2=0.0;
    
    double maxdist = dou_par[6];
    double nuis0 = dou_par[4];
    double nuis1 = dou_par[5];
    double par0 = dou_par[0];
    double par1 = dou_par[1];
    double par2 = dou_par[2];
    double par3 = dou_par[3];
    double REARTH = dou_par[8];
    
    
    int ncoord  = int_par[1];
    int cormod  = int_par[0];
    int type    = int_par[3];
    
    
    
    
    s1=nuis0+nuis1;
    
    
    for (j = 0; j < ncoord; j++) {
        if (   ((gid+j)!= j) && ((gid+j) < ncoord)   )
        {
            lags = dist(type,coordx[j],coordx[gid+j],coordy[j],coordy[gid+j],REARTH);
            
            if(lags<=maxdist){
            
                s12=nuis1*CorFct(cormod, lags, 0, par0,par1,par2,par3,0,0);
                det=pow(s1,2)-pow(s12,2);
            
                u=data[gid+j]-mean[gid+j];
                v=data[j]-mean[j];
                
                if(!isnan(u)&&!isnan(v) )
                {
                    u2=pow(u,2);
                    v2=pow(v,2);
                    sum+= (-log(2*M_PI)-log(det)+log(s1)+
                           (u2+v2)*(0.5/s1-s1/det)+2*s12*u*v/det)*weights;
                }
                
            }
            
        }
        
        else
            continue;
    }
    
    res[gid] = sum;
    
}


__kernel void Comp_Pair_Gauss2_OCL(__global const double *coordx,__global const double *coordy,__global const double *mean, __global const double *data, __global double *res,__global const int *int_par,__global const double *dou_par)
{
    
    int j, gid = get_global_id(0);
    
    double  corr=0.0, lags=0.0,weights=1.0, sum=0.0;
    //double det=0.0, u=0.0, u2=0.0, v=0.0, v2=0.0;
    
    double maxdist = dou_par[6];
    double nuis0 = dou_par[4];//nugget
    double nuis1 = dou_par[5]; //sill
    double par0 = dou_par[0];
    double par1 = dou_par[1];
    double par2 = dou_par[2];
    double par3 = dou_par[3];
    double REARTH = dou_par[8];
    
    
    int cormod  = int_par[0];
    int ncoord  = int_par[1];
    int type    = int_par[3];
    
    //s1=nuis0+nuis1;
    
    
    for (j = 0; j < ncoord; j++) {
        if (   ((gid+j)!= j) && ((gid+j) < ncoord)   )
        {
            lags = dist(type,coordx[j],coordx[gid+j],coordy[j],coordy[gid+j],REARTH);
            if(lags<=maxdist){
                corr=CorFct(cormod, lags, 0, par0,par1,par2,par3,0,0);
                if(!isnan(data[gid+j])&&!isnan(data[j]) )
                {
                    sum+=log_biv_Norm(corr,data[gid+j],data[j],mean[gid+j],mean[j],nuis1,nuis0)*weights;
                }}}
        
        else
            continue;
    }
    
    res[gid] = sum;
    
}


__kernel void Comp_Diff_Gauss2_OCL(__global const double *coordx,__global const double *coordy,__global const double *mean, __global const double *data, __global double *res,__global const int *int_par,__global const double *dou_par)
{
    
    int j, gid = get_global_id(0);
    double lags=0.0,weights=1.0, sum=0.0,vario=0.0;
    double det=0.0, u=0.0, u2=0.0, v=0.0, v2=0.0;
    
    double maxdist = dou_par[6];
    double nuis0 = dou_par[4];
    double nuis1 = dou_par[5];
    double par0 = dou_par[0];
    double par1 = dou_par[1];
    double par2 = dou_par[2];
    double par3 = dou_par[3];
    double REARTH = dou_par[8];
    
    int cormod      = int_par[0];
    int ncoord      = int_par[1];
    int weigthed    = int_par[2];
    int type        = int_par[3];
    
    
    for (j = 0; j < ncoord; j++) {
        if (   ((gid+j)!= j) && ((gid+j) < ncoord)   )
        {
            lags = dist(type,coordx[j],coordx[gid+j],coordy[j],coordy[gid+j],REARTH);
            if(lags<=maxdist){
                
                vario=Variogram(cormod,lags,0,nuis0,nuis1,par0,par1,par2,par3);
                u=data[gid+j];
                v=data[j];
                
                if(!isnan(u)&&!isnan(v) )
                {
                    if(weigthed) weights=CorFunBohman(lags,maxdist);
                    sum+=  -0.5*(log(2*M_PI)+log(vario)+
                                 pow(u-v,2)/(2*vario))*weights;
                }
                
            }
        }
        
        else
            continue;
    }
    //barrier(CLK_GLOBAL_MEM_FENCE);
    res[gid] = sum;
    
}



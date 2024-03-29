
#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

#define extern
#include "header.h"
#undef extern
//#include <stdlib.h> // for NULL



/* FIXME:
   Check these declarations against the C/Fortran source code.
*/

/* .C calls */



/********************** utility C calls ****************************************************/

extern void SetGlobalVar2 (int *nsite, int *times,
                           double *h,int *nn,  double *maxh,
                           double *u,int *tt,  double *maxu,
                           int *st,int *biv,int *one, int *two);
extern void SetGlobalVar(int *biv,double *coordx,double *coordy,double *coordt,int *grid,int *ia,
                         int *idx,int *ismal,int *ja,int *mem, int *nsite,int *nsitex,int *nsitey,
                         int *npair,double *radius,double *srange, double *sep,int *st, int *times,double *trange,
                         int *tap,int *tapmodel,int *tp,int *weighted, int *colidx,int *rowidx,
                     int *ns, int *NS, int *dyn);
extern void DeleteGlobalVar();
extern void DeleteGlobalVar2();
extern void GodambeMat(double *betas,int *biv,double *coordx, double *coordy, double *coordt, int *cormod, double *data, int *dst,
                       double *eps,int *flagcor, int *flagnuis, int *grid, int *like, double *mean,int *model,double *NN, int *nbetas,
                       int *npar, int *nparc,int *nparcT, double *parcor, double *nuis, double *score,
                       double *sensmat, int *spt,  int *type_lik, double *varimat,
                       int *vartype, double *winc, double *winstp,double *winct,double *winstp_t,int *weigthed, double *X,int *ns,int *NS);
extern void Maxima_Minima_dist(double *res,double *coordx,double *coordy,int *nsize,int *type_dist,double *radius);
extern void spectral_density(int *L,int *model,int *p, double *matrix ,double *matrix_out,
                             double *C, double *a, double *nu1,double *Cg, double *ag, double *nu1g);
extern void pairs(int *ncoords,double *data,double *coordx,double *coordy,double *numbins,double *bins,double *v0,double *v1,double *v2,double *maxdist);
extern void simu_on_coords(int *Ndim,int *Mcoords,int *Mu,double *coords,double *amatrix,
                           double *matrix_phi,double *matrix_u,double *matrix_out);

/********************** for variogrms computations  ****************************************************/

extern void Binned_Variogram_biv2(double *bins,double *coordx, double *coordy, double *coordt, double *data,
  int *cross_lbins, double *cross_moms, int *nbins,int *marg_lbins, double *marg_moms,int *ns, int *NS);
extern void Binned_Variogram_st2(double *bins, double *bint, double *coordx, double *coordy, double *coordt,double *data, int *lbins, int *lbinst,
       int *lbint, double *moms,double *momst, double *momt, int *nbins, int *nbint, int *ns,int *NS);
extern void Binned_Variogram2(double *bins, double *coordx, double *coordy, double *coordt,double *data, int *lbins, double *moms, int *nbins);
extern void Binned_Variogram2new(double *bins, int *np,double *data1, double *data2, double *vdist, int *lbins, double *moms, int *nbins,double *mm);
extern void Binned_Variogram_st2_dyn(double *bins, double *bint, double *coordx, double *coordy, double *coordt,double *data, int *lbins, int *lbinst,
                              int *lbint, double *moms,double *momst, double *momt, int *nbins, int *nbint, int *ns,int *NS);
extern void Cloud_Variogram2(double *bins,double *coordx, double *coordy, double *coordt, double *data, int *lbins, double *moms, int *nbins);
extern void LeastSquare_G(double *bins, double *bint, int *cormod, double *lbins, double *moms,
          int *nbins, int *nbint, double *nuis, double *par, double *res);
extern void WLeastSquare_G(double *bins, double *bint, int *cormod, double *lbins, double *moms,
           int *nbins, int *nbint, double *nuis, double *par, double *res);

extern void Binned_Variogram_22(double *bins, double *coordx, double *coordy, double *coordt,double *data, int *lbins, double *moms, int *nbins);

/********************** for distribution computations  ****************************************************/

extern void biv_unif_CopulaClayton_call(double *x,double *y,double *rho, double *nu, double *res);
extern void biv_unif_CopulaGauss_call(double *x,double *y,double *rho, double *res);
extern void corr_kuma_vec(double *rho,double *eta,double *gam,double *res, int *n);
extern void vpbnorm(int *cormod, double *h, double *u, int *nlags, int *nlagt,
                    double *nuis, double *par, double *rho, double *thr);

/*********************** for correlation computations ***************************************************/
extern void Corr_c(double *cc,double *coordx, double *coordy, double *coordt, int *cormod, int *grid, double *locx,  double *locy,
            int *ncoord, int *nloc,int *tloc,int *ns,int *NS,int *ntime, double *par, int *spt, int *biv, double *time,int *type, int *which,double *radius);
extern void Corr_c_bin(double *cc,double *coordx, double *coordy, double *coordt, int *cormod, int *grid, double *locx,  double *locy,int *ncoord, int *nloc,
                int *model,int *tloc,int *nn,int *n, int *ns,int *NS,int *ntime, double *mean,double *nuis, double *par, int *spt, int *biv, double *time,int *type, int *which,double *radius);
extern void Corr_c_tap(double *cc,double *cc_tap,double *coordx, double *coordy, double *coordt, int *cormod, int *cormodtap, int *grid, double *locx,  double *locy,
                double *mxd,double *mxt, int *ncoord, int *nloc, int *ns,int *NS,int*tloc,int *ntime, double *par, int *spt, int *biv, double *time,int *type,int *which,double *radius);
extern void VectCorrelation(double *rho, int *cormod, double *h, int *nlags, int *nlagt,double *mean,int *model,
                            double *nuis,double *par, double *u,int *N);
extern void VectCorrelation_biv(double *rho, double *vario,int *cormod, double *h, int *nlags, int *nlagt,double *mean,int *model,
                                double *nuis,double *par, double *u,int *N);
extern void CorrelationMat2(double *rho,double *coordx, double *coordy, double *coordt,  int *cormod,
 double *nuis, double *par,double *radius,int *ns, int *NS);
extern void CorrelationMat_dis2(double *rho,double *coordx, double *coordy, double *coordt,  int *cormod, double *mean,
                         int *nn,double *nuis, double *par,double *radius, int *ns, int *NS,int *model);
extern void CorrelationMat_st_dyn2(double *rho, double *coordx, double *coordy, double *coordt,int *cormod,
  double *nuis, double *par,double *radius, int *ns,int *NS);
extern void CorrelationMat_st_dyn_dis2(double *rho,double *coordx, double *coordy, double *coordt,  int *cormod,  double *mean,int *n,
                                double *nuis, double *par,double *radius, int *ns, int *NS, int *model);
extern void CorrelationMat_biv_dyn2(double *rho,double *coordx, double *coordy, double *coordt, int *cormod,  double *nuis,
  double *par,double *radius, int *ns,int *NS);
extern void CorrelationMat_biv_tap(double *rho, double *coordx, double *coordy, double *coordt,int *cormod,
 double *nuis, double *par,double *radius, int *ns,int *NS);
extern void DCorrelationMat_biv(int *cormod,double *coordx, double *coordy, double *coordt,double *drho,double *eps,int *flagcor,
      int *nparcor,double *parcor,double *rho);
extern void DCorrelationMat_biv2(int *cormod,double *coordx, double *coordy, double *coordt,double *drho,double *eps,int *flagcor,
      int *nparcor,double *parcor,double *rho);
extern void DCorrelationMat_biv_tap(int *cormod,double *coordx, double *coordy, double *coordt,double *drho,
            double *eps,int *flagcor,int *nparcor,double *parcor,double *rho);
extern void CorrelationMat_biv_skew_dyn2(double *rho,double *coordx, double *coordy, double *coordt, int *cormod,
 double *nuis, double *par,double *radius, int *ns,int *NS);
extern void CorrelationMat_tap(double *rho,double *coordx, double *coordy, double *coordt, int *cormod,  double *nuis, double *par,double *radius,
  int *ns, int *NS);
extern void CorrelationMat_dis_tap(double *rho,double *coordx, double *coordy, double *coordt, int *cormod,  double *nuis, double *par,double *radius,
  int *ns, int *NS, int *n1,int *n2, double *mu1,double *mu2,int  *model);
extern void CorrelationMat_st_tap(double *rho,double *coordx, double *coordy, double *coordt, int *cormod, double *nuis,
  double *par,double *radius, int *ns, int *NS);

/*********************** for bivariate composite likelihood ***************************************************/
extern void Comp_Pair_Gauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Diff_Gauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_WrapGauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_SinhGauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_SkewGauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gamma2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Weibull2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Kumaraswamy2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Kumaraswamy22mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Beta2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_LogGauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_PoisbinnegGauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_PoisbinGauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomnegGauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomnegLogi2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomnegGaussZINB2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomGauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomLogi2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomNNGauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomNNGauss_misp2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomNNLogi2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_LogLogistic2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Logistic2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Pois2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_PoisGamma2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_misp_PoisGamma2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_PoisZIP2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_misp_Pois2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_misp_PoisZIP2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_misp_SkewT2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_misp_T2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Tukeyhh2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Tukeyh2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_misp_Tukeygh2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_T2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_TWOPIECETukeyh2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_TWOPIECET2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_TWOPIECEGauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_TWOPIECEBIMODAL2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_WrapGauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_T_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Pois_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_PoisGamma_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_misp_PoisGamma_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_PoisZIP_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_misp_Pois_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_misp_PoisZIP_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Tukeyh_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Tukeyhh_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void  Comp_Pair_TWOPIECEGauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void  Comp_Pair_TWOPIECETukeyh_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void  Comp_Pair_TWOPIECET_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_PoisbinGauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_PoisbinnegGauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomGauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomLogi_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomNNGauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomNNLogi_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomnegGauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomnegLogi_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomnegGaussZINB_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Diff_Gauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_SkewGauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_SinhGauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gamma_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Kumaraswamy_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Kumaraswamy2_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Beta_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Weibull_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_LogGauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_LogLogistic_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Logistic_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_TWOPIECEBIMODAL_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_biv2mem(int *cormod, double *data1,double *data2,int *NN,
    double *par, int *weigthed,double *res,double *mean1,double *mean2,
    double *nuis, int *local,int *GPU);
extern void Comp_Pair_SkewGauss_biv2mem(int *cormod, double *data1,double *data2,int *NN,
    double *par, int *weigthed,double *res,double *mean1,double *mean2,
    double *nuis, int *local,int *GPU);
extern void Comp_Pair_GaussGCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_BetaGCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Beta2GCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_KumaraswamyGCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Kumaraswamy2GCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_GaussCCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_BetaCCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Beta2CCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_KumaraswamyCCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Kumaraswamy2CCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_misp_T_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);

/*********************** for conditional composite likelihood ***************************************************/

extern void Comp_Cond_Gauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                         double *nuis, int *local,int *GPU);
extern void Comp_Cond_Tukeyh2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                          double *nuis, int *local,int *GPU);
extern void Comp_Cond_Tukeyhh2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                           double *nuis, int *local,int *GPU);
extern void Comp_Cond_SkewGauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                             double *nuis, int *local,int *GPU);
extern void Comp_Cond_T2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                     double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gauss_misp_T2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gauss_misp_SkewT2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                    double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gauss_misp_Tukeygh2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                      double *nuis, int *local,int *GPU);
extern void Comp_Cond_SinhGauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                             double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gamma2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                         double *nuis, int *local,int *GPU);
extern void Comp_Cond_Weibull2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                           double *nuis, int *local,int *GPU);
extern void Comp_Cond_LogGauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                            double *nuis, int *local,int *GPU);
extern void Comp_Cond_Beta2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                        double *nuis, int *local,int *GPU);
extern void Comp_Cond_Kumaraswamy2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                               double *nuis, int *local,int *GPU);
extern void Comp_Cond_Kumaraswamy22mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gauss_misp_Pois2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomNNGauss_misp2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                     double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gauss_misp_PoisGamma2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                        double *nuis, int *local,int *GPU);
extern void Comp_Cond_PoisGamma2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                             double *nuis, int *local,int *GPU);
extern void Comp_Cond_Pois2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                        double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomGauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                              double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomLogi2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                             double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomNNGauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomNNLogi2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                               double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomnegGauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                 double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomnegLogi2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                double *nuis, int *local,int *GPU);
extern void Comp_Cond_TWOPIECETukeyh2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                  double *nuis, int *local,int *GPU);
extern void Comp_Cond_TWOPIECET2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                             double *nuis, int *local,int *GPU);
extern void Comp_Cond_TWOPIECEGauss2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                 double *nuis, int *local,int *GPU);
extern void Comp_Cond_TWOPIECEBIMODAL2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomnegGaussZINB2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                     double *nuis, int *local,int *GPU);
extern void Comp_Cond_PoisZIP2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                           double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gauss_misp_PoisZIP2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                      double *nuis, int *local,int *GPU);
extern void Comp_Cond_LogLogistic2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                               double *nuis, int *local,int *GPU);
extern void Comp_Cond_Logistic2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                            double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                            double *nuis, int *local,int *GPU);
extern void Comp_Cond_T_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                        double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gauss_misp_T_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Cond_Tukeyh_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                             double *nuis, int *local,int *GPU);
extern void Comp_Cond_Tukeyhh_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                              double *nuis, int *local,int *GPU);
extern void Comp_Cond_SkewGauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                double *nuis, int *local,int *GPU);
extern void Comp_Cond_SinhGauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gamma_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                            double *nuis, int *local,int *GPU);
extern void Comp_Cond_Weibull_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                              double *nuis, int *local,int *GPU);
extern void Comp_Cond_LogGauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                               double *nuis, int *local,int *GPU);
extern void Comp_Cond_Beta_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                           double *nuis, int *local,int *GPU);
extern void Comp_Cond_Kumaraswamy_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                  double *nuis, int *local,int *GPU);
extern void Comp_Cond_Kumaraswamy2_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gauss_misp_Pois_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                      double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gauss_misp_PoisGamma_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                           double *nuis, int *local,int *GPU);
extern void Comp_Cond_PoisGamma_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                double *nuis, int *local,int *GPU);
extern void Comp_Cond_Pois_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                           double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomGauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                 double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomLogi_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomNNGauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomNNLogi_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                  double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomnegGauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                    double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomnegLogi_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Cond_TWOPIECETukeyh_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                     double *nuis, int *local,int *GPU);
extern void Comp_Cond_TWOPIECET_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                 double *nuis, int *local,int *GPU);
extern void Comp_Cond_TWOPIECEGauss_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                    double *nuis, int *local,int *GPU);
extern void Comp_Cond_TWOPIECEBIMODAL_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                      double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomnegGaussZINB_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                        double *nuis, int *local,int *GPU);
extern void Comp_Cond_PoisZIP_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                              double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gauss_misp_PoisZIP_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                         double *nuis, int *local,int *GPU);
extern void Comp_Cond_LogLogistic_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                  double *nuis, int *local,int *GPU);
extern void Comp_Cond_Logistic_st2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                               double *nuis, int *local,int *GPU);
extern void Comp_Cond_GaussGCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                             double *nuis, int *local,int *GPU);
extern void Comp_Cond_BetaGCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                            double *nuis, int *local,int *GPU);
extern void Comp_Cond_Beta2GCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                             double *nuis, int *local,int *GPU);
extern void Comp_Cond_KumaraswamyGCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Cond_Kumaraswamy2GCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                    double *nuis, int *local,int *GPU);
extern void Comp_Cond_GaussCCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                             double *nuis, int *local,int *GPU);
extern void Comp_Cond_BetaCCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                            double *nuis, int *local,int *GPU);
extern void Comp_Cond_Beta2CCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                             double *nuis, int *local,int *GPU);
extern void Comp_Cond_KumaraswamyCCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Cond_Kumaraswamy2CCop2mem(int *cormod, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                    double *nuis, int *local,int *GPU);

/********************** spatial anisotropic  marginal************************************/
extern void Comp_Pair_Gauss2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                               double *nuis, int *local,int *GPU);
extern void Comp_Diff_Gauss2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                               double *nuis, int *local,int *GPU);
extern void Comp_Pair_WrapGauss2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Pair_SinhGauss2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Pair_SkewGauss2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gamma2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                               double *nuis, int *local,int *GPU);
extern void Comp_Pair_Weibull2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Kumaraswamy2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                     double *nuis, int *local,int *GPU);
extern void Comp_Pair_Kumaraswamy22mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                      double *nuis, int *local,int *GPU);
extern void Comp_Pair_Beta2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                              double *nuis, int *local,int *GPU);
extern void Comp_Pair_LogGauss2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                  double *nuis, int *local,int *GPU);
extern void Comp_Pair_PoisbinnegGauss2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                         double *nuis, int *local,int *GPU);
extern void Comp_Pair_PoisbinGauss2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                      double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomnegGauss2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                       double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomnegLogi2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                      double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomnegGaussZINB2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                           double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomGauss2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                    double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomLogi2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomNNGauss2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                      double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomNNGauss_misp2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                           double *nuis, int *local,int *GPU);
extern void Comp_Pair_BinomNNLogi2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                     double *nuis, int *local,int *GPU);
extern void Comp_Pair_LogLogistic2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                     double *nuis, int *local,int *GPU);
extern void Comp_Pair_Logistic2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                  double *nuis, int *local,int *GPU);
extern void Comp_Pair_Pois2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                              double *nuis, int *local,int *GPU);
extern void Comp_Pair_PoisGamma2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_misp_PoisGamma2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                              double *nuis, int *local,int *GPU);
extern void Comp_Pair_PoisZIP2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_misp_Pois2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                         double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_misp_PoisZIP2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                            double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_misp_SkewT2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                          double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_misp_T2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                      double *nuis, int *local,int *GPU);
extern void Comp_Pair_Tukeyhh2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                 double *nuis, int *local,int *GPU);
extern void Comp_Pair_Tukeyh2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                double *nuis, int *local,int *GPU);
extern void Comp_Pair_Gauss_misp_Tukeygh2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                            double *nuis, int *local,int *GPU);
extern void Comp_Pair_T2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                           double *nuis, int *local,int *GPU);
extern void Comp_Pair_TWOPIECETukeyh2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                        double *nuis, int *local,int *GPU);
extern void Comp_Pair_TWOPIECET2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Pair_TWOPIECEGauss2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                       double *nuis, int *local,int *GPU);
extern void Comp_Pair_TWOPIECEBIMODAL2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                         double *nuis, int *local,int *GPU);
extern void Comp_Pair_GaussGCop2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Pair_BetaGCop2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                  double *nuis, int *local,int *GPU);
extern void Comp_Pair_Beta2GCop2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Pair_KumaraswamyGCop2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                         double *nuis, int *local,int *GPU);
extern void Comp_Pair_Kumaraswamy2GCop2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                          double *nuis, int *local,int *GPU);
extern void Comp_Pair_GaussCCop2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Pair_BetaCCop2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                  double *nuis, int *local,int *GPU);
extern void Comp_Pair_Beta2CCop2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                   double *nuis, int *local,int *GPU);
extern void Comp_Pair_KumaraswamyCCop2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                         double *nuis, int *local,int *GPU);
extern void Comp_Pair_Kumaraswamy2CCop2mem_aniso(int *cormod, double *coord1,double *coord2, double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
                                          double *nuis, int *local,int *GPU);

/********************** spatial anisotropic conditional ************************************/
extern void Comp_Cond_Gauss2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_Tukeyhh2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_SkewGauss2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_T2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gauss_misp_T2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gauss_misp_SkewT2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gauss_misp_Tukeygh2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_SinhGauss2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gamma2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_Weibull2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_LogGauss2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_Beta2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_Kumaraswamy2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_Kumaraswamy22mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gauss_misp_Pois2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomNNGauss_misp2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gauss_misp_PoisGamma2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_PoisGamma2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_Pois2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomGauss2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomLogi2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomNNGauss2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomNNLogi2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomnegGauss2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomnegLogi2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_TWOPIECETukeyh2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_TWOPIECET2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_TWOPIECEGauss2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_TWOPIECEBIMODAL2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_BinomnegGaussZINB2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_PoisZIP2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_Gauss_misp_PoisZIP2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_LogLogistic2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
extern void Comp_Cond_Logistic2mem_aniso(int *cormod, double *coord1, double *coord2,double *data1,double *data2,int *NN,
 double *par, int *weigthed, double *res,double *mean1,double *mean2,
 double *nuis, int *local,int *GPU);
/*********************************************************************************************/
extern void CorrelationMat_st_dis_tap(double *rho,double *coordx, double *coordy, double *coordt, int *cormod,  double *nuis, double *par,double *radius,
                               int *ns, int *NS, int *n1,int *n2, double *mu1,double *mu2,int  *model);

extern void hypergeo_call(double *a,double *b,double *c,double *x, double *res);

extern void hyperg_call(double *a,double *b,double *x,double *res);
extern void biv_pois_call(double *corr,int *r, int *t, double *mean_i, double *mean_j,double *res);
extern void appellF4_call(double *a,double *b,double *c,double *d,double *x,double *y, double *res);
extern void biv_gamma_call(double *corr,double *zi,double *zj,double *mui, double *muj, double *shape, double *res);

extern void biv_binomneg_call(int *NN,int *u, int *v, double *p01, double *p10,double *p11,double *res);

extern void biv_binom_call(int *NN,int *u, int *v, double *p01, double *p10,double *p11,double *res);

extern void matrix_temp(int *N ,double *matrix, double *l1 ,double *l2 ,double *v11 ,double *v21,double *v12,double *v22);

extern void vector_to_select(int *N, double *matrix);


extern void lgnd (int *lmax,double *x, double *p);
extern void qnorm55_call(double *p, double *mu, double *sigma, int *lower_tail, int *log_p, double *res);


extern void integr_kuma(double *x, int n, void *ex);

static const R_CMethodDef CEntries[] = {
    {"integr_kuma",               (DL_FUNC) &integr_kuma,              3},
    {"lgnd",                      (DL_FUNC) &lgnd,                     3},
    {"qnorm55_call",              (DL_FUNC) &qnorm55_call,             6},
    {"matrix_temp",               (DL_FUNC) &matrix_temp,              8},
    {"vector_to_select",          (DL_FUNC) &vector_to_select,         2},
    {"biv_binomneg_call",         (DL_FUNC) &biv_binomneg_call,        7},
    {"biv_binom_call",            (DL_FUNC) &biv_binom_call,           7},
    {"biv_gamma_call",            (DL_FUNC) &biv_gamma_call,           7},
    {"hyperg_call",               (DL_FUNC) &hyperg_call,              4},
    {"biv_pois_call",             (DL_FUNC) &biv_pois_call,            6},
    {"appellF4_call",             (DL_FUNC) &appellF4_call,            7},
    {"hypergeo_call",             (DL_FUNC) &hypergeo_call,            5},
    {"CorrelationMat_st_dis_tap", (DL_FUNC) &CorrelationMat_st_dis_tap,15},
    {"CorrelationMat2",             (DL_FUNC) &CorrelationMat2,             10},
    {"CorrelationMat_dis2",         (DL_FUNC) &CorrelationMat_dis2,         13},
    {"CorrelationMat_st_dyn2",      (DL_FUNC) &CorrelationMat_st_dyn2,      10},
    {"CorrelationMat_biv_dyn2",     (DL_FUNC) &CorrelationMat_biv_dyn2,     10},
    {"CorrelationMat_st_dyn_dis2",  (DL_FUNC) &CorrelationMat_st_dyn_dis2,  13},
    {"CorrelationMat_biv_tap",      (DL_FUNC) &CorrelationMat_biv_tap,      10},
    {"DCorrelationMat_biv",         (DL_FUNC) &DCorrelationMat_biv,         10},
    {"DCorrelationMat_biv2",        (DL_FUNC) &DCorrelationMat_biv2,        10},
    {"DCorrelationMat_biv_tap",     (DL_FUNC) &DCorrelationMat_biv_tap,     10},
    {"CorrelationMat_biv_skew_dyn2",(DL_FUNC) &CorrelationMat_biv_skew_dyn2,10},
    {"CorrelationMat_tap",          (DL_FUNC) &CorrelationMat_tap,          10},
    {"CorrelationMat_dis_tap",      (DL_FUNC) &CorrelationMat_dis_tap,      15},
    {"CorrelationMat_st_tap",       (DL_FUNC) &CorrelationMat_st_tap,       10},
    {"Binned_Variogram2",           (DL_FUNC) &Binned_Variogram2,            8},
    {"Binned_Variogram2new",        (DL_FUNC) &Binned_Variogram2new,         9},
    {"Binned_Variogram_biv2",       (DL_FUNC) &Binned_Variogram_biv2,       12},
    {"Binned_Variogram_st2",        (DL_FUNC) &Binned_Variogram_st2,        16},
    {"Binned_Variogram_st2_dyn",    (DL_FUNC) &Binned_Variogram_st2_dyn,    16},
    {"Cloud_Variogram2",            (DL_FUNC) &Cloud_Variogram2,             8},
    {"LeastSquare_G",               (DL_FUNC) &LeastSquare_G,               10},
    {"WLeastSquare_G",              (DL_FUNC) &WLeastSquare_G,              10},
    {"Binned_Variogram_22",         (DL_FUNC) &Binned_Variogram_22,          8},
    {"biv_unif_CopulaClayton_call", (DL_FUNC) &biv_unif_CopulaClayton_call,  5},
    {"biv_unif_CopulaGauss_call",   (DL_FUNC) &biv_unif_CopulaGauss_call,    4},
    {"Corr_c",                      (DL_FUNC) &Corr_c,                      21},
    {"Corr_c_bin",                  (DL_FUNC) &Corr_c_bin,                  26},
    {"Corr_c_tap",                  (DL_FUNC) &Corr_c_tap,                  25},
    {"corr_kuma_vec",               (DL_FUNC) &corr_kuma_vec,                5},
    {"DeleteGlobalVar",             (DL_FUNC) &DeleteGlobalVar,              0},
    {"DeleteGlobalVar2",            (DL_FUNC) &DeleteGlobalVar2,             0},
    {"GodambeMat",                  (DL_FUNC) &GodambeMat,                  36},
    {"Maxima_Minima_dist",          (DL_FUNC) &Maxima_Minima_dist,           6},
    {"pairs",                       (DL_FUNC) &pairs,                       10},
    {"SetGlobalVar2",               (DL_FUNC) &SetGlobalVar2,               12},
    {"SetGlobalVar",               (DL_FUNC) &SetGlobalVar,               29},
    {"simu_on_coords",              (DL_FUNC) &simu_on_coords,               8},
    {"spectral_density",            (DL_FUNC) &spectral_density,            11},
    {"VectCorrelation",             (DL_FUNC) &VectCorrelation,             11},
    {"VectCorrelation_biv",         (DL_FUNC) &VectCorrelation_biv,         12},
    {"vpbnorm",                     (DL_FUNC) &vpbnorm,                      9},

  /*********************** for pairwise composite likelihood ***************************************************/
    
    {"Comp_Pair_Gauss2mem",         (DL_FUNC) &Comp_Pair_Gauss2mem,         12},
    {"Comp_Pair_Tukeyh2mem",         (DL_FUNC) &Comp_Pair_Tukeyh2mem,         12},
    {"Comp_Pair_Tukeyhh2mem",         (DL_FUNC) &Comp_Pair_Tukeyhh2mem,         12},
    {"Comp_Pair_SkewGauss2mem",         (DL_FUNC) &Comp_Pair_SkewGauss2mem,         12},
    {"Comp_Pair_T2mem",         (DL_FUNC) &Comp_Pair_T2mem,         12},
    {"Comp_Pair_Gauss_misp_T2mem",         (DL_FUNC) &Comp_Pair_Gauss_misp_T2mem,         12},
    {"Comp_Pair_Gauss_misp_SkewT2mem",         (DL_FUNC) &Comp_Pair_Gauss_misp_SkewT2mem,         12},
    {"Comp_Pair_Gauss_misp_Tukeygh2mem",         (DL_FUNC) &Comp_Pair_Gauss_misp_Tukeygh2mem,         12},
    {"Comp_Pair_SinhGauss2mem",         (DL_FUNC) &Comp_Pair_SinhGauss2mem,         12},
    {"Comp_Pair_Gamma2mem",         (DL_FUNC) &Comp_Pair_Gamma2mem,         12},
    {"Comp_Pair_Weibull2mem",         (DL_FUNC) &Comp_Pair_Weibull2mem,         12},
    {"Comp_Pair_LogGauss2mem",         (DL_FUNC) &Comp_Pair_LogGauss2mem,         12},
    {"Comp_Pair_Beta2mem",         (DL_FUNC) &Comp_Pair_Beta2mem,         12},
    {"Comp_Pair_Kumaraswamy2mem",         (DL_FUNC) &Comp_Pair_Kumaraswamy2mem,         12},
    {"Comp_Pair_Kumaraswamy22mem",         (DL_FUNC) &Comp_Pair_Kumaraswamy22mem,         12},
    {"Comp_Pair_Gauss_misp_Pois2mem",         (DL_FUNC) &Comp_Pair_Gauss_misp_Pois2mem,         12},
    {"Comp_Pair_BinomNNGauss_misp2mem",         (DL_FUNC) &Comp_Pair_BinomNNGauss_misp2mem,         12},
    {"Comp_Pair_Gauss_misp_PoisGamma2mem",         (DL_FUNC) &Comp_Pair_Gauss_misp_PoisGamma2mem,         12},
    {"Comp_Pair_PoisGamma2mem",         (DL_FUNC) &Comp_Pair_PoisGamma2mem,         12},
    {"Comp_Pair_Pois2mem",         (DL_FUNC) &Comp_Pair_Pois2mem,         12},
    {"Comp_Pair_BinomGauss2mem",         (DL_FUNC) &Comp_Pair_BinomGauss2mem,         12},
    {"Comp_Pair_BinomLogi2mem",         (DL_FUNC) &Comp_Pair_BinomLogi2mem,         12},
    {"Comp_Pair_BinomNNGauss2mem",         (DL_FUNC) &Comp_Pair_BinomNNGauss2mem,         12},
    {"Comp_Pair_BinomNNLogi2mem",         (DL_FUNC) &Comp_Pair_BinomNNLogi2mem,         12},
    {"Comp_Pair_BinomnegGauss2mem",         (DL_FUNC) &Comp_Pair_BinomnegGauss2mem,         12},
    {"Comp_Pair_TWOPIECETukeyh2mem",         (DL_FUNC) &Comp_Pair_TWOPIECETukeyh2mem,         12},
    {"Comp_Pair_TWOPIECET2mem",         (DL_FUNC) &Comp_Pair_TWOPIECET2mem,         12},
    {"Comp_Pair_TWOPIECEGauss2mem",         (DL_FUNC) &Comp_Pair_TWOPIECEGauss2mem,         12},
    {"Comp_Pair_TWOPIECEBIMODAL2mem",         (DL_FUNC) &Comp_Pair_TWOPIECEBIMODAL2mem,         12},
    {"Comp_Pair_BinomnegGaussZINB2mem",         (DL_FUNC) &Comp_Pair_BinomnegGaussZINB2mem,         12},
    {"Comp_Pair_PoisZIP2mem",         (DL_FUNC) &Comp_Pair_PoisZIP2mem,         12},
    {"Comp_Pair_Gauss_misp_PoisZIP2mem",         (DL_FUNC) &Comp_Pair_Gauss_misp_PoisZIP2mem,         12},
    {"Comp_Pair_Gauss_st2mem",         (DL_FUNC) &Comp_Pair_Gauss_st2mem,         12},
    {"Comp_Pair_T_st2mem",         (DL_FUNC) &Comp_Pair_T_st2mem,         12},
    {"Comp_Pair_Gauss_misp_T_st2mem",         (DL_FUNC) &Comp_Pair_Gauss_misp_T_st2mem,         12},
    {"Comp_Pair_Tukeyh_st2mem",         (DL_FUNC) &Comp_Pair_Tukeyh_st2mem,         12},
    {"Comp_Pair_Tukeyhh_st2mem",         (DL_FUNC) &Comp_Pair_Tukeyhh_st2mem,         12},
    {"Comp_Pair_SkewGauss_st2mem",         (DL_FUNC) &Comp_Pair_SkewGauss_st2mem,         12},
    {"Comp_Pair_SinhGauss_st2mem",         (DL_FUNC) &Comp_Pair_SinhGauss_st2mem,         12},
    {"Comp_Pair_Gamma_st2mem",         (DL_FUNC) &Comp_Pair_Gamma_st2mem,         12},
    {"Comp_Pair_Weibull_st2mem",         (DL_FUNC) &Comp_Pair_Weibull_st2mem,         12},
    {"Comp_Pair_LogGauss_st2mem",         (DL_FUNC) &Comp_Pair_LogGauss_st2mem,         12},
    {"Comp_Pair_Beta_st2mem",         (DL_FUNC) &Comp_Pair_Beta_st2mem,         12},
    {"Comp_Pair_Kumaraswamy_st2mem",         (DL_FUNC) &Comp_Pair_Kumaraswamy_st2mem,         12},//85
    {"Comp_Pair_Kumaraswamy2_st2mem",         (DL_FUNC) &Comp_Pair_Kumaraswamy2_st2mem,         12},
    {"Comp_Pair_Gauss_misp_Pois_st2mem",         (DL_FUNC) &Comp_Pair_Gauss_misp_Pois_st2mem,         12},
    {"Comp_Pair_Gauss_misp_PoisGamma_st2mem",         (DL_FUNC) &Comp_Pair_Gauss_misp_PoisGamma_st2mem,         12},
    {"Comp_Pair_PoisGamma_st2mem",         (DL_FUNC) &Comp_Pair_PoisGamma_st2mem,         12},
    {"Comp_Pair_Pois_st2mem",         (DL_FUNC) &Comp_Pair_Pois_st2mem,         12},
    {"Comp_Pair_BinomGauss_st2mem",         (DL_FUNC) &Comp_Pair_BinomGauss_st2mem,         12},
    {"Comp_Pair_BinomLogi_st2mem",         (DL_FUNC) &Comp_Pair_BinomLogi_st2mem,         12},
    {"Comp_Pair_BinomNNGauss_st2mem",         (DL_FUNC) &Comp_Pair_BinomNNGauss_st2mem,         12},
    {"Comp_Pair_BinomNNLogi_st2mem",         (DL_FUNC) &Comp_Pair_BinomNNLogi_st2mem,         12},
    {"Comp_Pair_BinomnegGauss_st2mem",         (DL_FUNC) &Comp_Pair_BinomnegGauss_st2mem,         12},
    {"Comp_Pair_BinomnegLogi_st2mem",         (DL_FUNC) &Comp_Pair_BinomnegLogi_st2mem,         12},//96
    {"Comp_Pair_TWOPIECETukeyh_st2mem",         (DL_FUNC) &Comp_Pair_TWOPIECETukeyh_st2mem,         12},
    {"Comp_Pair_TWOPIECET_st2mem",         (DL_FUNC) &Comp_Pair_TWOPIECET_st2mem,         12},
    {"Comp_Pair_TWOPIECEGauss_st2mem",         (DL_FUNC) &Comp_Pair_TWOPIECEGauss_st2mem,         12},
    {"Comp_Pair_TWOPIECEBIMODAL_st2mem",         (DL_FUNC) &Comp_Pair_TWOPIECEBIMODAL_st2mem,         12},
    {"Comp_Pair_BinomnegGaussZINB_st2mem",         (DL_FUNC) &Comp_Pair_BinomnegGaussZINB_st2mem,         12},
    {"Comp_Pair_PoisZIP_st2mem",         (DL_FUNC) &Comp_Pair_PoisZIP_st2mem,         12},
    {"Comp_Pair_Gauss_misp_PoisZIP_st2mem",         (DL_FUNC) &Comp_Pair_Gauss_misp_PoisZIP_st2mem,         12},
    {"Comp_Pair_LogLogistic_st2mem",         (DL_FUNC) &Comp_Pair_LogLogistic_st2mem,         12},
    {"Comp_Pair_Logistic_st2mem",         (DL_FUNC) &Comp_Pair_Logistic_st2mem,         12},
    {"Comp_Pair_GaussGCop2mem",         (DL_FUNC) &Comp_Pair_GaussGCop2mem,         12},
    {"Comp_Pair_BetaGCop2mem",         (DL_FUNC) &Comp_Pair_BetaGCop2mem,         12},
    {"Comp_Pair_Beta2GCop2mem",         (DL_FUNC) &Comp_Pair_Beta2GCop2mem,         12},
    {"Comp_Pair_KumaraswamyGCop2mem",         (DL_FUNC) &Comp_Pair_KumaraswamyGCop2mem,         12},
    {"Comp_Pair_Kumaraswamy2GCop2mem",         (DL_FUNC) &Comp_Pair_Kumaraswamy2GCop2mem,         12},//90
    {"Comp_Pair_GaussCCop2mem",         (DL_FUNC) &Comp_Pair_GaussCCop2mem,         12},
    {"Comp_Pair_BetaCCop2mem",         (DL_FUNC) &Comp_Pair_BetaCCop2mem,         12},
    {"Comp_Pair_Beta2CCop2mem",         (DL_FUNC) &Comp_Pair_Beta2CCop2mem,         12},//93
  
    /*********************** for conditional composite likelihood ***************************************************/
    
    {"Comp_Cond_Gauss2mem",         (DL_FUNC) &Comp_Cond_Gauss2mem,         12},
    {"Comp_Cond_Tukeyh2mem",         (DL_FUNC) &Comp_Cond_Tukeyh2mem,         12},
    {"Comp_Cond_Tukeyhh2mem",         (DL_FUNC) &Comp_Cond_Tukeyhh2mem,         12},
    {"Comp_Cond_SkewGauss2mem",         (DL_FUNC) &Comp_Cond_SkewGauss2mem,         12},
    {"Comp_Cond_T2mem",         (DL_FUNC) &Comp_Cond_T2mem,         12},
    {"Comp_Cond_Gauss_misp_T2mem",         (DL_FUNC) &Comp_Cond_Gauss_misp_T2mem,         12},
    {"Comp_Cond_Gauss_misp_SkewT2mem",         (DL_FUNC) &Comp_Cond_Gauss_misp_SkewT2mem,         12},
    {"Comp_Cond_Gauss_misp_Tukeygh2mem",         (DL_FUNC) &Comp_Cond_Gauss_misp_Tukeygh2mem,         12},
    {"Comp_Cond_SinhGauss2mem",         (DL_FUNC) &Comp_Cond_SinhGauss2mem,         12},
    {"Comp_Cond_Gamma2mem",         (DL_FUNC) &Comp_Cond_Gamma2mem,         12},
    {"Comp_Cond_Weibull2mem",         (DL_FUNC) &Comp_Cond_Weibull2mem,         12},
    {"Comp_Cond_LogGauss2mem",         (DL_FUNC) &Comp_Cond_LogGauss2mem,         12},
    {"Comp_Cond_Beta2mem",         (DL_FUNC) &Comp_Cond_Beta2mem,         12},
    {"Comp_Cond_Kumaraswamy2mem",         (DL_FUNC) &Comp_Cond_Kumaraswamy2mem,         12},
    {"Comp_Cond_Kumaraswamy22mem",         (DL_FUNC) &Comp_Cond_Kumaraswamy22mem,         12},
    {"Comp_Cond_Gauss_misp_Pois2mem",         (DL_FUNC) &Comp_Cond_Gauss_misp_Pois2mem,         12},
    {"Comp_Cond_BinomNNGauss_misp2mem",         (DL_FUNC) &Comp_Cond_BinomNNGauss_misp2mem,         12},
    {"Comp_Cond_Gauss_misp_PoisGamma2mem",         (DL_FUNC) &Comp_Cond_Gauss_misp_PoisGamma2mem,         12},
    {"Comp_Cond_PoisGamma2mem",         (DL_FUNC) &Comp_Cond_PoisGamma2mem,         12},
    {"Comp_Cond_Pois2mem",         (DL_FUNC) &Comp_Cond_Pois2mem,         12},
    {"Comp_Cond_BinomGauss2mem",         (DL_FUNC) &Comp_Cond_BinomGauss2mem,         12},
    {"Comp_Cond_BinomLogi2mem",         (DL_FUNC) &Comp_Cond_BinomLogi2mem,         12},
    {"Comp_Cond_BinomNNGauss2mem",         (DL_FUNC) &Comp_Cond_BinomNNGauss2mem,         12},
    {"Comp_Cond_BinomNNLogi2mem",         (DL_FUNC) &Comp_Cond_BinomNNLogi2mem,         12},
    {"Comp_Cond_BinomnegGauss2mem",         (DL_FUNC) &Comp_Cond_BinomnegGauss2mem,         12},
    {"Comp_Cond_TWOPIECETukeyh2mem",         (DL_FUNC) &Comp_Cond_TWOPIECETukeyh2mem,         12},
    {"Comp_Cond_TWOPIECET2mem",         (DL_FUNC) &Comp_Cond_TWOPIECET2mem,         12},
    {"Comp_Cond_TWOPIECEGauss2mem",         (DL_FUNC) &Comp_Cond_TWOPIECEGauss2mem,         12},
    {"Comp_Cond_TWOPIECEBIMODAL2mem",         (DL_FUNC) &Comp_Cond_TWOPIECEBIMODAL2mem,         12},
    {"Comp_Cond_BinomnegGaussZINB2mem",         (DL_FUNC) &Comp_Cond_BinomnegGaussZINB2mem,         12},
    {"Comp_Cond_PoisZIP2mem",         (DL_FUNC) &Comp_Cond_PoisZIP2mem,         12},
    {"Comp_Cond_Gauss_misp_PoisZIP2mem",         (DL_FUNC) &Comp_Cond_Gauss_misp_PoisZIP2mem,         12},
    {"Comp_Cond_Gauss_st2mem",         (DL_FUNC) &Comp_Cond_Gauss_st2mem,         12},
    {"Comp_Cond_T_st2mem",         (DL_FUNC) &Comp_Cond_T_st2mem,         12},
    {"Comp_Cond_Gauss_misp_T_st2mem",         (DL_FUNC) &Comp_Cond_Gauss_misp_T_st2mem,         12},
    {"Comp_Cond_Tukeyh_st2mem",         (DL_FUNC) &Comp_Cond_Tukeyh_st2mem,         12},
    {"Comp_Cond_Tukeyhh_st2mem",         (DL_FUNC) &Comp_Cond_Tukeyhh_st2mem,         12},
    {"Comp_Cond_SkewGauss_st2mem",         (DL_FUNC) &Comp_Cond_SkewGauss_st2mem,         12},
    {"Comp_Cond_SinhGauss_st2mem",         (DL_FUNC) &Comp_Cond_SinhGauss_st2mem,         12},
    {"Comp_Cond_Gamma_st2mem",         (DL_FUNC) &Comp_Cond_Gamma_st2mem,         12},
    {"Comp_Cond_Weibull_st2mem",         (DL_FUNC) &Comp_Cond_Weibull_st2mem,         12},
    {"Comp_Cond_LogGauss_st2mem",         (DL_FUNC) &Comp_Cond_LogGauss_st2mem,         12},
    {"Comp_Cond_Beta_st2mem",         (DL_FUNC) &Comp_Cond_Beta_st2mem,         12},
    {"Comp_Cond_Kumaraswamy_st2mem",         (DL_FUNC) &Comp_Cond_Kumaraswamy_st2mem,         12},//85
    {"Comp_Cond_Kumaraswamy2_st2mem",         (DL_FUNC) &Comp_Cond_Kumaraswamy2_st2mem,         12},
    {"Comp_Cond_Gauss_misp_Pois_st2mem",         (DL_FUNC) &Comp_Cond_Gauss_misp_Pois_st2mem,         12},
    {"Comp_Cond_Gauss_misp_PoisGamma_st2mem",         (DL_FUNC) &Comp_Cond_Gauss_misp_PoisGamma_st2mem,         12},
    {"Comp_Cond_PoisGamma_st2mem",         (DL_FUNC) &Comp_Cond_PoisGamma_st2mem,         12},
    {"Comp_Cond_Pois_st2mem",         (DL_FUNC) &Comp_Cond_Pois_st2mem,         12},
    {"Comp_Cond_BinomGauss_st2mem",         (DL_FUNC) &Comp_Cond_BinomGauss_st2mem,         12},
    {"Comp_Cond_BinomLogi_st2mem",         (DL_FUNC) &Comp_Cond_BinomLogi_st2mem,         12},
    {"Comp_Cond_BinomNNGauss_st2mem",         (DL_FUNC) &Comp_Cond_BinomNNGauss_st2mem,         12},
    {"Comp_Cond_BinomNNLogi_st2mem",         (DL_FUNC) &Comp_Cond_BinomNNLogi_st2mem,         12},
    {"Comp_Cond_BinomnegGauss_st2mem",         (DL_FUNC) &Comp_Cond_BinomnegGauss_st2mem,         12},
    {"Comp_Cond_BinomnegLogi_st2mem",         (DL_FUNC) &Comp_Cond_BinomnegLogi_st2mem,         12},//96
    {"Comp_Cond_TWOPIECETukeyh_st2mem",         (DL_FUNC) &Comp_Cond_TWOPIECETukeyh_st2mem,         12},
    {"Comp_Cond_TWOPIECET_st2mem",         (DL_FUNC) &Comp_Cond_TWOPIECET_st2mem,         12},
    {"Comp_Cond_TWOPIECEGauss_st2mem",         (DL_FUNC) &Comp_Cond_TWOPIECEGauss_st2mem,         12},
    {"Comp_Cond_TWOPIECEBIMODAL_st2mem",         (DL_FUNC) &Comp_Cond_TWOPIECEBIMODAL_st2mem,         12},
    {"Comp_Cond_BinomnegGaussZINB_st2mem",         (DL_FUNC) &Comp_Cond_BinomnegGaussZINB_st2mem,         12},
    {"Comp_Cond_PoisZIP_st2mem",         (DL_FUNC) &Comp_Cond_PoisZIP_st2mem,         12},
    {"Comp_Cond_Gauss_misp_PoisZIP_st2mem",         (DL_FUNC) &Comp_Cond_Gauss_misp_PoisZIP_st2mem,         12},
    {"Comp_Cond_LogLogistic_st2mem",         (DL_FUNC) &Comp_Cond_LogLogistic_st2mem,         12},
    {"Comp_Cond_Logistic_st2mem",         (DL_FUNC) &Comp_Cond_Logistic_st2mem,         12},
    {"Comp_Cond_GaussGCop2mem",         (DL_FUNC) &Comp_Cond_GaussGCop2mem,         12},
    {"Comp_Cond_BetaGCop2mem",         (DL_FUNC) &Comp_Cond_BetaGCop2mem,         12},
    {"Comp_Cond_Beta2GCop2mem",         (DL_FUNC) &Comp_Cond_Beta2GCop2mem,         12},
    {"Comp_Cond_KumaraswamyGCop2mem",         (DL_FUNC) &Comp_Cond_KumaraswamyGCop2mem,         12},
    {"Comp_Cond_Kumaraswamy2GCop2mem",         (DL_FUNC) &Comp_Cond_Kumaraswamy2GCop2mem,         12},//90
    {"Comp_Cond_GaussCCop2mem",         (DL_FUNC) &Comp_Cond_GaussCCop2mem,         12},
    {"Comp_Cond_BetaCCop2mem",         (DL_FUNC) &Comp_Cond_BetaCCop2mem,         12},
    {"Comp_Cond_Beta2CCop2mem",         (DL_FUNC) &Comp_Cond_Beta2CCop2mem,         12},//93
    
    /********************** spatial anisotropic +************************************/
    {"Comp_Pair_Gauss2mem_aniso",         (DL_FUNC) &Comp_Pair_Gauss2mem_aniso,         14},
    {"Comp_Diff_Gauss2mem_aniso",         (DL_FUNC) &Comp_Diff_Gauss2mem_aniso,         14},
    {"Comp_Pair_WrapGauss2mem_aniso",         (DL_FUNC) &Comp_Pair_WrapGauss2mem_aniso,         14},
    {"Comp_Pair_SinhGauss2mem_aniso",         (DL_FUNC) &Comp_Pair_SinhGauss2mem_aniso,         14},
    {"Comp_Pair_SkewGauss2mem_aniso",         (DL_FUNC) &Comp_Pair_SkewGauss2mem_aniso,         14},
    {"Comp_Pair_Gamma2mem_aniso",         (DL_FUNC) &Comp_Pair_Gamma2mem_aniso,         14},
    {"Comp_Pair_Weibull2mem_aniso",         (DL_FUNC) &Comp_Pair_Weibull2mem_aniso,         14},
    {"Comp_Pair_Kumaraswamy22mem_aniso",         (DL_FUNC) &Comp_Pair_Kumaraswamy22mem_aniso,         14},
    {"Comp_Pair_Beta2mem_aniso",         (DL_FUNC) &Comp_Pair_Beta2mem_aniso,         14},
    {"Comp_Pair_LogGauss2mem_aniso",         (DL_FUNC) &Comp_Pair_LogGauss2mem_aniso,         14},
    {"Comp_Pair_PoisbinnegGauss2mem_aniso",         (DL_FUNC) &Comp_Pair_PoisbinnegGauss2mem_aniso,         14},
    {"Comp_Pair_PoisbinGauss2mem_aniso",         (DL_FUNC) &Comp_Pair_PoisbinGauss2mem_aniso,         14},
    {"Comp_Pair_BinomnegGauss2mem_aniso",         (DL_FUNC) &Comp_Pair_BinomnegGauss2mem_aniso,         14},//109
    {"Comp_Pair_BinomnegLogi2mem_aniso",         (DL_FUNC) &Comp_Pair_BinomnegLogi2mem_aniso,         14},
    {"Comp_Pair_BinomnegGaussZINB2mem_aniso",         (DL_FUNC) &Comp_Pair_BinomnegGaussZINB2mem_aniso,         14},
    {"Comp_Pair_BinomGauss2mem_aniso",         (DL_FUNC) &Comp_Pair_BinomGauss2mem_aniso,         14},//112
    {"Comp_Pair_BinomLogi2mem_aniso",         (DL_FUNC) &Comp_Pair_BinomLogi2mem_aniso,         14},
    {"Comp_Pair_BinomNNGauss2mem_aniso",         (DL_FUNC) &Comp_Pair_BinomNNGauss2mem_aniso,         14},
    {"Comp_Pair_BinomNNGauss_misp2mem_aniso",         (DL_FUNC) &Comp_Pair_BinomNNGauss_misp2mem_aniso,         14},
    {"Comp_Pair_BinomNNLogi2mem_aniso",         (DL_FUNC) &Comp_Pair_BinomNNLogi2mem_aniso,         14},
    {"Comp_Pair_LogLogistic2mem_aniso",         (DL_FUNC) &Comp_Pair_LogLogistic2mem_aniso,         14},
    {"Comp_Pair_Logistic2mem_aniso",         (DL_FUNC) &Comp_Pair_Logistic2mem_aniso,         14},
    {"Comp_Pair_Pois2mem_aniso",         (DL_FUNC) &Comp_Pair_Pois2mem_aniso,         14},
    {"Comp_Pair_PoisGamma2mem_aniso",         (DL_FUNC) &Comp_Pair_PoisGamma2mem_aniso,         14},
    {"Comp_Pair_Gauss_misp_PoisGamma2mem_aniso",         (DL_FUNC) &Comp_Pair_Gauss_misp_PoisGamma2mem_aniso,         14},
    {"Comp_Pair_PoisZIP2mem_aniso",         (DL_FUNC) &Comp_Pair_PoisZIP2mem_aniso,         14},
    {"Comp_Pair_Gauss_misp_Pois2mem_aniso",         (DL_FUNC) &Comp_Pair_Gauss_misp_Pois2mem_aniso,         14},
    {"Comp_Pair_Gauss_misp_PoisZIP2mem_aniso",         (DL_FUNC) &Comp_Pair_Gauss_misp_PoisZIP2mem_aniso,         14},
    {"Comp_Pair_Gauss_misp_SkewT2mem_aniso",         (DL_FUNC) &Comp_Pair_Gauss_misp_SkewT2mem_aniso,         14},
    {"Comp_Pair_Gauss_misp_T2mem_aniso",         (DL_FUNC) &Comp_Pair_Gauss_misp_T2mem_aniso,         14},
    {"Comp_Pair_Tukeyhh2mem_aniso",         (DL_FUNC) &Comp_Pair_Tukeyhh2mem_aniso,         14},
    {"Comp_Pair_Tukeyh2mem_aniso",         (DL_FUNC) &Comp_Pair_Tukeyh2mem_aniso,         14},
    {"Comp_Pair_Gauss_misp_Tukeygh2mem_aniso",         (DL_FUNC) &Comp_Pair_Gauss_misp_Tukeygh2mem_aniso,         14},
    {"Comp_Pair_T2mem_aniso",         (DL_FUNC) &Comp_Pair_T2mem_aniso,         14},
    {"Comp_Pair_TWOPIECETukeyh2mem_aniso",         (DL_FUNC) &Comp_Pair_TWOPIECETukeyh2mem_aniso,         14},
    {"Comp_Pair_TWOPIECET2mem_aniso",         (DL_FUNC) &Comp_Pair_TWOPIECET2mem_aniso,         14},
    {"Comp_Pair_TWOPIECEGauss2mem_aniso",         (DL_FUNC) &Comp_Pair_TWOPIECEGauss2mem_aniso,         14},
    {"Comp_Pair_TWOPIECEBIMODAL2mem_aniso",         (DL_FUNC) &Comp_Pair_TWOPIECEBIMODAL2mem_aniso,         14},
    {"Comp_Pair_GaussGCop2mem_aniso",         (DL_FUNC) &Comp_Pair_GaussGCop2mem_aniso,         14},
    {"Comp_Pair_BetaGCop2mem_aniso",         (DL_FUNC) &Comp_Pair_BetaGCop2mem_aniso,         14},
    {"Comp_Pair_Beta2GCop2mem_aniso",         (DL_FUNC) &Comp_Pair_Beta2GCop2mem_aniso,         14},
    {"Comp_Pair_KumaraswamyGCop2mem_aniso",         (DL_FUNC) &Comp_Pair_KumaraswamyGCop2mem_aniso,         14},
    {"Comp_Pair_Kumaraswamy2GCop2mem_aniso",         (DL_FUNC) &Comp_Pair_Kumaraswamy2GCop2mem_aniso,         14},
    {"Comp_Pair_GaussCCop2mem_aniso",         (DL_FUNC) &Comp_Pair_GaussCCop2mem_aniso,         14},
    {"Comp_Pair_Beta2CCop2mem_aniso",         (DL_FUNC) &Comp_Pair_Beta2CCop2mem_aniso,         14},
    {"Comp_Pair_KumaraswamyCCop2mem_aniso",         (DL_FUNC) &Comp_Pair_KumaraswamyCCop2mem_aniso,         14},
    {"Comp_Pair_Kumaraswamy2CCop2mem_aniso",         (DL_FUNC) &Comp_Pair_Kumaraswamy2CCop2mem_aniso,         14},
    {"Comp_Cond_Gauss2mem_aniso",         (DL_FUNC) &Comp_Cond_Gauss2mem_aniso,         14},
    {"Comp_Cond_Tukeyhh2mem_aniso",         (DL_FUNC) &Comp_Cond_Tukeyhh2mem_aniso,         14},
    {"Comp_Cond_SkewGauss2mem_aniso",         (DL_FUNC) &Comp_Cond_SkewGauss2mem_aniso,         14},
    {"Comp_Cond_T2mem_aniso",         (DL_FUNC) &Comp_Cond_T2mem_aniso,         14},
    {"Comp_Cond_Gauss_misp_T2mem_aniso",         (DL_FUNC) &Comp_Cond_Gauss_misp_T2mem_aniso,         14},
    {"Comp_Cond_Gauss_misp_SkewT2mem_aniso",         (DL_FUNC) &Comp_Cond_Gauss_misp_SkewT2mem_aniso,         14},
    {"Comp_Cond_SinhGauss2mem_aniso",         (DL_FUNC) &Comp_Cond_SinhGauss2mem_aniso,         14},
    {"Comp_Cond_Gamma2mem_aniso",         (DL_FUNC) & Comp_Cond_Gamma2mem_aniso,         14},
    {"Comp_Cond_Weibull2mem_aniso",         (DL_FUNC) &Comp_Cond_Weibull2mem_aniso,         14},
    {"Comp_Cond_LogGauss2mem_aniso",         (DL_FUNC) &Comp_Cond_LogGauss2mem_aniso,         14},
    {"Comp_Cond_Beta2mem_aniso",         (DL_FUNC) &Comp_Cond_Beta2mem_aniso,         14},
    {"Comp_Cond_Kumaraswamy2mem_aniso",         (DL_FUNC) &Comp_Cond_Kumaraswamy2mem_aniso,         14},
    {"Comp_Cond_Gauss_misp_Pois2mem_aniso",         (DL_FUNC) &Comp_Cond_Gauss_misp_Pois2mem_aniso,         14},
    {"Comp_Cond_BinomNNGauss_misp2mem_aniso",         (DL_FUNC) &Comp_Cond_BinomNNGauss_misp2mem_aniso,         14},
    {"Comp_Cond_Gauss_misp_PoisGamma2mem_aniso",         (DL_FUNC) &Comp_Cond_Gauss_misp_PoisGamma2mem_aniso,         14},
    {"Comp_Cond_PoisGamma2mem_aniso",         (DL_FUNC) &Comp_Cond_PoisGamma2mem_aniso,         14},
    {"Comp_Cond_Pois2mem_aniso",         (DL_FUNC) &Comp_Cond_Pois2mem_aniso,         14},
    {"Comp_Cond_BinomGauss2mem_aniso",         (DL_FUNC) &Comp_Cond_BinomGauss2mem_aniso,         14},
    {"Comp_Cond_BinomLogi2mem_aniso",         (DL_FUNC) &Comp_Cond_BinomLogi2mem_aniso,         14},
    {"Comp_Cond_BinomNNGauss2mem_aniso",         (DL_FUNC) &Comp_Cond_BinomNNGauss2mem_aniso,         14},
    {"Comp_Cond_BinomNNLogi2mem_aniso",         (DL_FUNC) & Comp_Cond_BinomNNLogi2mem_aniso,         14},
    {"Comp_Cond_BinomnegGauss2mem_aniso",         (DL_FUNC) &Comp_Cond_BinomnegGauss2mem_aniso,         14},
    {"Comp_Cond_TWOPIECETukeyh2mem_aniso",         (DL_FUNC) &Comp_Cond_TWOPIECETukeyh2mem_aniso,         14},
    {"Comp_Cond_TWOPIECET2mem_aniso",         (DL_FUNC) &Comp_Cond_TWOPIECET2mem_aniso,         14},
    {"Comp_Cond_TWOPIECEGauss2mem_aniso",         (DL_FUNC) &Comp_Cond_TWOPIECEGauss2mem_aniso,         14},
    {"Comp_Cond_TWOPIECEBIMODAL2mem_aniso",         (DL_FUNC) &Comp_Cond_TWOPIECEBIMODAL2mem_aniso,         14},
    {"Comp_Cond_BinomnegGaussZINB2mem_aniso",         (DL_FUNC) &Comp_Cond_BinomnegGaussZINB2mem_aniso,         14},
    {"Comp_Cond_PoisZIP2mem_aniso",         (DL_FUNC) &Comp_Cond_PoisZIP2mem_aniso,         14},
    {"Comp_Cond_LogLogistic2mem_aniso",         (DL_FUNC) &Comp_Cond_LogLogistic2mem_aniso,         14},
    {"Comp_Cond_LogLogistic2mem",         (DL_FUNC) &Comp_Cond_LogLogistic2mem,         12},
    {"Comp_Cond_Logistic2mem_aniso",         (DL_FUNC) &Comp_Cond_Logistic2mem_aniso,         14},
    {"Comp_Pair_Gauss_biv2mem",         (DL_FUNC) &Comp_Pair_Gauss_biv2mem,         12},
    {NULL, NULL, 0}
};

void R_init_GeoModels(DllInfo *dll)
{
    R_registerRoutines(dll, CEntries, NULL, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
void R_unload_GeoModels(DllInfo *info) {
  // just to avoid warning from compiler
}

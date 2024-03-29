####################################################
### File name: GeoCv.r
####################################################

GeoCV=function(fit, K=100, estimation=FALSE, n.fold=0.05, local=FALSE,neighb=NULL,
                          maxdist=NULL,maxtime=NULL,
                          sparse=FALSE, which=1,seed=1)
{

if(n.fold>0.99||n.fold<0.01) stop("n.fold must be beween 0.01 and 0.99")
print("Cross-validation  kriging can be time consuming ...")
if(ncol(fit$X)==1) {X=Xloc=NULL}
mae=rmse=lscore=crps=NULL
space_dyn=FALSE

if(is.list(fit$data)) space_dyn=TRUE
spacetime<-CheckST(CkCorrModel(fit$corrmodel))
bivariate<-CheckBiv(CkCorrModel(fit$corrmodel))
K=round(K)
if(K<2) stop("K must be grater or equal  to 2")
if(K>10000) stop("K is  too large")
if(bivariate)
   {if(!(which==1||which==2))
          stop("which must be 1 or 2")}
if(local) if(is.null(maxdist)&&is.null(neighb)) stop("maxdist or neighb are required
          for local kriging")
i=1
print(paste("Starting iteration from 1 to",K," ..."))
space=!spacetime&&!bivariate

dtp=pred=list()


model1=fit$model
if(fit$missp)  ### misspecification
 {if(fit$model=="StudentT")     model1="Gaussian_misp_StudentT"
  if(fit$model=="Poisson")      model1="Gaussian_misp_Poisson"
  if(fit$model=="PoissonZIP")   model1="Gaussian_misp_PoissonZIP"
  if(fit$model=="SkewStudentT") model1="Gaussian_misp_SkewStudenT"
  if(fit$model=="Tukeygh")      model1="Gaussian_misp_Tukeygh"
 }

############################################################
########### spatial case ###################################
############################################################
if(space)
{
N=length(fit$data)
coords=cbind(fit$coordx,fit$coordy)
data=fit$data

set.seed(round(seed))
pb <- txtProgressBar(min = 0, max = K, style = 3)
while(i<=K){
Sys.sleep(0.1)
sel_data = sample(1:N,round(N*(1-n.fold)))  
# data and coord used for prediction

if(!is.null(X)) {
                X=fit$X[sel_data,]
                Xloc=fit$X[-sel_data,]
                }
# data and coord to predict
data_to_pred  = fit$data[-sel_data]

dtp[[i]]=data_to_pred

#param=as.list(c(fit$param,fit$fixed))
param=append(fit$param,fit$fixed)
if(estimation) {
          fit_s= GeoFit2(data=fit$data[sel_data],coordx=coords[sel_data,],corrmodel=fit$corrmodel,X=X,
                            likelihood=fit$likelihood,type=fit$type,grid=fit$grid,
                            copula=fit$copula,anisopars=fit$anisopars,est.aniso=fit$est.aniso,
                            model=model1,radius=fit$radius,n=fit$n,
                            local=fit$local,GPU=fit$GPU,
                           maxdist=fit$maxdist, neighb=fit$neighb,distance=fit$distance,
                            optimizer=fit$optimizer, lower=fit$lower,upper=fit$upper,
                           # start=as.list(fit$param),fixed=as.list(fit$fixed))
          start=fit$param,fixed=fit$fixed)
            

            if(!is.null(fit$anisopars))   
                   {   fit_s$param$angle=NULL;fit_s$param$ratio=NULL; fit_s$fixed$angle=NULL;fit_s$fixed$ratio=NULL}

            param=append(fit_s$param,fit_s$fixed)
           
              }
 
if(!local) pr=GeoKrig(data=fit$data[sel_data], coordx=coords[sel_data,],  
	            corrmodel=fit$corrmodel, distance=fit$distance,grid=fit$grid,loc=coords[-sel_data,], #ok
	            model=fit$model, n=fit$n, mse=TRUE,#ok
              param=param, anisopars=fit_s$anisopars,
               radius=fit$radius, sparse=sparse, X=X,Xloc=Xloc) #ok

if(local) pr=GeoKrigloc(data=fit$data[sel_data], coordx=coords[sel_data,],  
              corrmodel=fit$corrmodel, distance=fit$distance,grid=fit$grid,loc=coords[-sel_data,], #ok
              model=fit$model, n=fit$n, mse=TRUE,#ok
              neighb=neighb,maxdist=maxdist,
              param=param, anisopars=fit_s$anisopars,
              radius=fit$radius, sparse=sparse, X=X,Xloc=Xloc) #ok

pred[[i]]=as.numeric(pr$pred)
err=data_to_pred-pr$pred  

sqrtvv=sqrt(pr$mse)
std=err/sqrtvv

N2=length(err)
rmse=c(rmse,sqrt(sum(err^2)/N2))
mae= c(mae,      sum(abs(err))/N2)
lscore=c(lscore, 0.5*sum(std^2+log(2*pi*sqrtvv))/N2 )
crps=c(crps,sum( sqrtvv*( std*(2*pnorm(std)-1 ) +2*pnorm(std)-1/sqrt(pi)))/N2)
setTxtProgressBar(pb, i)
i=i+1} ##end while
close(pb)
}
############################################################
########### spatial bivariate case ###################################
############################################################
if(bivariate)
{
ns=fit$ns
if(space_dyn) {data1=fit$data[[1]]
               data2=fit$data[[2]]
               coords1=fit$coordx_dyn[[1]]
               coords2=fit$coordx_dyn[[2]]
               }
if(!space_dyn) {data1=fit$data[1,];data2=fit$data[2,]
                coords=cbind(fit$coordx,fit$coordy)
                coords1=coords; coords2=coords
               }
if(!is.null(X)) {
                if(!is.list(fit$X)){X1=fit$X[1:ns[1],];X2=fit$X[(ns[1]+1):(ns[1]+ns[2]),]}
                if( is.list(fit$X)){X1=fit$X[[1]];X2=fit$X[[2]]}
                }
set.seed(round(seed))

pb <- txtProgressBar(min = 0, max = K, style = 3)
#param=as.list(c(fit$param,fit$fixed))
param=append(fit$param,fit$fixed)
while(i<=K){
Sys.sleep(0.1)
#######################################	
if(which==1) {
	          sel_data = sample(1:ns[1],round(ns[1]*(1-n.fold))) 
              cc1= coords1[sel_data,]
              loc_to_pred   = coords1[-sel_data,]
              d1   = data1[sel_data]
              data_to_pred  = data1[-sel_data]
              if(!is.null(X)) { X=rbind(X1[sel_data,],X2); Xloc=rbind(X1[-sel_data,],X2)}
datanew=list();datanew[[1]]=d1;datanew[[2]]=data2;
coordsnew=list();coordsnew[[1]]=cc1;coordsnew[[2]]=coords2;
          }
if(which==2) {
	          sel_data = sample(1:ns[2],round(ns[2]*(1-n.fold))) 
              cc2= coords2[sel_data,]
              loc_to_pred   = coords2[-sel_data,]
              d2   = data2[sel_data]
              data_to_pred  = data2[-sel_data]
              if(!is.null(X)) {X=rbind(X1,X2[sel_data,]); Xloc=rbind(X1,X2[-sel_data,])}
datanew=list();datanew[[1]]=data1;datanew[[2]]=d2;
coordsnew=list();coordsnew[[1]]=coords1;coordsnew[[2]]=cc2;
            }
dtp[[i]]=data_to_pred

if(estimation) {
          fit_s= GeoFit(data=fit$data[sel_data],coordx=coords[sel_data,],corrmodel=fit$corrmodel,X=X,
                            likelihood=fit$likelihood,c,type=fit$type,grid=fit$grid,
                            model=model1,radius=fit$radius,n=fit$n,
                               copula=fit$copula,
                             local=fit$local,GPU=fit$GPU,
                           maxdist=fit$maxdist, neighb=fit$neighb,distance=fit$distance,
                            optimizer=fit$optimizer, lower=fit$lower,upper=fit$upper,
                           # start=as.list(fit$param),fixed=as.list(fit$fixed))
                               start=fit$param,fixed=fit$fixed)

             if(!is.null(fit$anisopars))   
                   {   fit_s$param$angle=NULL;fit_s$param$ratio=NULL;
                       fit_s$fixed$angle=NULL;fit_s$fixed$ratio=NULL
                        }

            param=append(fit_s$param,fit_s$fixed)
              }
#####################################
if(!local) pr=GeoKrig(data=datanew, coordx=NULL,   coordt=NULL, coordx_dyn=coordsnew,  #ok
	       corrmodel=fit$corrmodel, distance=fit$distance,grid=fit$grid,loc=loc_to_pred, #ok
	          model=fit$model, n=fit$n, mse=TRUE,#ok
           param=param, 
           radius=fit$radius, sparse=sparse,   time=NULL, 
             which=which, X=X,Xloc=Xloc) #ok  

if(local) pr=GeoKrigloc(data=datanew, coordx=NULL,   coordt=NULL, coordx_dyn=coordsnew,  #ok
         corrmodel=fit$corrmodel, distance=fit$distance,grid=fit$grid,loc=loc_to_pred, #ok
            model=fit$model, n=fit$n, mse=TRUE,#ok
           neighb=neighb, maxdist=maxdist,
           param=param, 
           radius=fit$radius, sparse=sparse,   time=NULL, 
             which=which, X=X,Xloc=Xloc) #ok  

pred[[i]]=as.numeric(pr$pred)
err=data_to_pred-pr$pred  

sqrtvv=sqrt(pr$mse)
std=err/sqrtvv

   
N2=length(err)
rmse=c(rmse,sqrt(sum(err^2)/N2))
mae= c(mae,      sum(abs(err))/N2)
lscore=c(lscore, 0.5*sum(std^2+log(2*pi*sqrtvv))/N2 )
crps=c(crps,sum( sqrtvv*( std*(2*pnorm(std)-1 ) +2*pnorm(std)-1/sqrt(pi)))/N2)
#print(i)
i=i+1
setTxtProgressBar(pb, i)
}               
close(pb) 
} ## end bivariate
############################################################
########### spatio temporal case ###########################
############################################################

if(spacetime)
{

coords=cbind(fit$coordx,fit$coordy)
ns=fit$ns
coordt=fit$coordt
T=length(coordt)
NT=sum(ns)
if(is.null(X)) X=rep(1,NT)

NS=cumsum(ns)
NS=c(c(0,NS)[-(length(ns)+1)],NT)

if(!space_dyn){
data_tot=NULL
for(k in 1:T) 
data_tot=rbind(data_tot,cbind(rep(coordt[k],ns[k]),fit$data[k,]))
data_tot=cbind(coords,data_tot,fit$X)
}


set.seed(round(seed))

pb <- txtProgressBar(min = 0, max = K, style = 3)
while(i<=K){
Sys.sleep(0.1)
####################################### 
sel_data = sample(1:NT,round(NT*(1-n.fold))) 
data_sel=data_tot[sel_data,]
data_to_pred=data_tot[-sel_data,]

data_sel_ord=data_sel[order(data_sel[,3]),]
data_to_pred_ord=data_to_pred[order(data_to_pred[,3]),]

DD=ncol(data_sel_ord)
k=1 ; coordx_dynnew=Xnew=datanew=list()

utt=unique(data_sel_ord[,3])
for(k in 1:length(utt) ){
ss=data_sel_ord[data_sel_ord[,3]==utt[k],]

datanew[[k]]=as.vector((ss[,4]))
coordx_dynnew[[k]]=as.matrix(ss[,1:2])

#if(!is.null(X)) { 
Xnew[[k]]=as.vector(ss[,5:DD])
#}
}
if(is.vector(Xnew[[1]])) Xnew=NULL

#param=as.list(c(fit$param,fit$fixed))
param=append(fit$param,fit$fixed)
if(estimation) {

          fit_s= GeoFit2(data=datanew,coordx_dyn=coordx_dynnew,coordt=utt,
                            corrmodel=fit$corrmodel,X=Xnew,
                            likelihood=fit$likelihood,type=fit$type,grid=fit$grid,
                            copula=fit$copula, anisopars=fit$anisopars,est.aniso=fit$est.aniso,
                            model=model1,radius=fit$radius,n=fit$n,
                            local=fit$local,GPU=fit$GPU,
                            maxdist=fit$maxdist, neighb=fit$neighb,maxtime=fit$maxtime,distance=fit$distance,
                            optimizer=fit$optimizer, lower=fit$lower,upper=fit$upper,
                            start=fit$param,fixed=fit$fixed)
                            #start=as.list(fit$param),fixed=as.list(fit$fixed))
       
            if(!is.null(fit$anisopars))   
                   {   fit_s$param$angle=NULL;fit_s$param$ratio=NULL; fit_s$fixed$angle=NULL;fit_s$fixed$ratio=NULL}
           param=append(fit_s$param,fit_s$fixed)
              }
             

##### to finish---



err=c(data_to_pred)-c(pr$pred)  

sqrtvv=sqrt(pr$mse)
std=err/sqrtvv


N2=length(err)
rmse=c(rmse,sqrt(sum(err^2)/N2))
mae= c(mae,      sum(abs(err))/N2)
lscore=c(lscore, 0.5*sum(std^2+log(2*pi*sqrtvv))/N2 )
crps=c(crps,sum( sqrtvv*( std*(2*pnorm(std)-1 ) +2*pnorm(std)-1/sqrt(pi)))/N2)
#print(i)
i=i+1
setTxtProgressBar(pb, i)
}               
close(pb) 
} ## end spacetime

return(list(rmse=rmse,mae=mae,crps=crps,lscore=lscore,predicted=pred,data_to_pred=dtp))
}

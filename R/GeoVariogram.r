####################################################
### Authors:  Moreno Bevilacqua, Víctor Morales Oñate.
### Email: moreno.bevilacqua@uv.cl, victor.morales@uv.cl
### Instituto de Estadistica
### Universidad de Valparaiso
### File name: GeoVariogram.r
### Description:
### This file contains a set of procedures in order
### to estimate the empirical variogram
### Last change: 28/03/2013.
####################################################

### Procedures are in alphabetical order.

GeoVariogram <- function(data, coordx, coordy=NULL, coordt=NULL, coordx_dyn=NULL,cloud=FALSE, distance="Eucl",
                       grid=FALSE, maxdist=NULL, maxtime=NULL, numbins=NULL,
                       radius=6371, type='variogram',bivariate=FALSE)
  {
    call <- match.call()
    corrmodel <- 'gauss'
    ### Check the parameters given in input:
    if(is.null(type))
      type <- 'variogram'
    # Set the type of model:
    if(type=='variogram'){
        model <- 'Gaussian'
        fname <- 'Binned_Variogram'}
    ##if(type=="lorelogram"){
    ##    model <- "BinaryGauss"
    ##    fname <- "Binned_Lorelogram"}
    
    # Checks if its a spatial or spatial-temporal random field:
    if(bivariate) coordt=c(0,1)
    if(!is.null(coordt))
      if(is.numeric(coordt))
        if(length(coordt)>1) corrmodel <- 'gneiting'
    # Checks the input:
    checkinput <- CkInput(coordx, coordy, coordt, coordx_dyn, corrmodel, data, distance, "Fitting", NULL, grid,
                             'None', maxdist, maxtime, model,NULL, 'Nelder-Mead', NULL,
                             radius,  NULL, NULL,NULL, 'GeoWLS', FALSE, 'SubSamp', FALSE,NULL)
                             

    # Checks if there are errors in the input:
    if(!is.null(checkinput$error))
      stop(checkinput$error)
    ### START -- Specific checks of the Empirical Variogram:
    if(!is.null(cloud) & !is.logical(cloud))
      stop('insert a logical value (TRUE/FALSE) for the cloud parameter\n')
    
    if(!is.null(numbins) & !is.integer(numbins))
      if(numbins < 0)
        stop('insert a positive integer value for the number of bins\n')
    if(is.null(numbins))
      numbins <- 13
    ### END -- Specific checks of the Empirical Variogram
    if(bivariate)  corrmodel <- 'Bi_matern_sep'
  

    n=1
    initparam <- StartParam(coordx, coordy, coordt,coordx_dyn, corrmodel, data,distance, "Fitting",
                           NULL, grid, 'None', maxdist,
                           maxtime, model, n, NULL, NULL, FALSE, radius, 
                           NULL, NULL, NULL, 'GeoWLS', 'GeoWLS', FALSE,
                           'SubSamp', FALSE, 1, 1,1,1,NULL)

    coordx=initparam$coordx;coordy=initparam$coordy;coordt=initparam$coordt                 
    # Checks if there are inconsistences:
    if(!is.null(initparam$error))
      stop(initparam$error)
    numvario <- numbins-1
    if(cloud){
        numbins <- numvario <- initparam$numpairs
        fname <- 'Cloud_Variogram'}
    ### Estimation of the empirical spatial or spatial-temporal variogram:
    bins <- double(numbins) # spatial bins
    moments <- double(numvario) # vector of spatial moments
    lenbins <- integer(numvario) # vector of spatial bin sizes
    bint <- NULL
    lenbinst <- NULL
    lenbint <- NULL
    variogramst <- NULL
    variogramt <- NULL  
  #***********************************************************************************************#
  #***********************************************************************************************#
  #***********************************************************************************************#
    if(initparam$bivariate){
               n_var=initparam$numtime
               spacetime_dyn=FALSE
               if(!is.null(coordx_dyn)) spacetime_dyn=TRUE
               ns=initparam$ns
               NS=cumsum(ns)
               if(!spacetime_dyn){
                                  data=c(t(data))
                                  coordx=rep(coordx,n_var)
                                  coordy=rep(coordy,n_var)
                         }
               if(spacetime_dyn) data=unlist(data)
               NS=c(0,NS)[-(length(ns)+1)]
               moments_marg<-double(n_var*numvario)   # vect of square differences for each component (n_var) 11  e 22
               lenbins_marg<-integer(n_var*numvario)  #
               moments_cross<-double(0.5*n_var*(n_var-1)*numvario)  # vect of square differences for cross components (12)
               lenbins_cross<-integer(0.5*n_var*(n_var-1)*numvario) #
               DEV=.C("Binned_Variogram_biv2", bins=bins, as.double(coordx),as.double(coordy),as.double(coordt),as.double(data),
               lenbins_cross=lenbins_cross, moments_cross=moments_cross, as.integer(numbins),lenbins_marg=lenbins_marg,
               moments_marg=moments_marg,as.integer(ns),as.integer(NS),
               PACKAGE='GeoModels', DUP = TRUE, NAOK=TRUE)
               bins=DEV$bins
               lenbins_cross=DEV$lenbins_cross
               moments_cross=DEV$moments_cross
               lenbins_marg=DEV$lenbins_marg
               moments_marg=DEV$moments_marg
               m_11=moments_marg[1:numvario];m_22=moments_marg[(numvario+1):(2*numvario)];m_12=moments_cross[1:numvario];
               l_11=lenbins_marg[1:numvario];l_22=lenbins_marg[(numvario+1):(2*numvario)];l_12=lenbins_cross[1:numvario];
               indbin_marg <- l_11>0;indbin_cross <- l_12>0  #
               bins<- bins[indbin_marg]
               numbins <-length(bins)
               m_11 <- m_11[indbin_marg];m_22 <- m_22[indbin_marg];l_11 <- l_11[indbin_marg];l_22 <- l_22[indbin_marg]
               m_12 <- m_12[indbin_cross];l_12 <- l_12[indbin_cross]
               variograms_11 <- m_11/l_11;variograms_22 <- m_22/l_22
               variograms_12 <- m_12/l_12   
               centers <-   bins[1:(numbins[1]-1)]+diff(bins)/2
               lenbins=rbind(l_11,l_22);lenbinst=l_12
               variograms=rbind(variograms_11,variograms_22) 
               variogramst=variograms_12
    }
  #***********************************************************************************************#
  #***********************************************************************************************#
  #***********************************************************************************************#
    if(initparam$spacetime){
      numtime=initparam$numtime
      spacetime_dyn=FALSE
      if(!is.null(coordx_dyn)) spacetime_dyn=TRUE
      ns=initparam$ns
      NS=cumsum(ns)
      numbint <- initparam$numtime-1 # number of temporal bins
      bint <- double(numbint)        # temporal bins
      momentt <- double(numbint)     # vector of temporal moments
      lenbint <- integer(numbint)    # vector of temporal bin sizes
      numbinst <- numvario*numbint   # number of spatial-temporal bins
      binst <- double(numbinst)      # spatial-temporal bins
      momentst <- double(numbinst)   # vector of spatial-temporal moments
      lenbinst <- integer(numbinst)  # vector of spatial-temporal bin sizes
      if(cloud) fname <- 'Cloud_Variogram_st' else fname <- 'Binned_Variogram_st'
      #if(type=="lorelogram") fname <- "Binned_Lorelogram_st"
      if(initparam$bivariate)  fname <- 'Binned_Variogram_biv'
      if(grid)     {a=expand.grid(coordx,coordy);coordx=a[,1];coordy=a[,2]; }
      else{
      if(!spacetime_dyn){
                                  data=c(t(data))
                                  coordx=rep(coordx,numtime)
                                  coordy=rep(coordy,numtime)
                         }
      if(spacetime_dyn) data=unlist(data)
      }
         NS=c(0,NS)[-(length(ns)+1)]
      fname <- paste(fname,"2",sep="") 
      # Compute the spatial-temporal moments:
      EV=.C(fname, bins=bins, bint=bint,  as.double(coordx),as.double(coordy),as.double(coordt),as.double((data)),
           lenbins=lenbins,lenbinst=lenbinst,lenbint=lenbint,moments=moments,momentst=momentst,momentt=momentt,
           as.integer(numbins), as.integer(numbint),as.integer(ns),as.integer(NS), PACKAGE='GeoModels', DUP = TRUE, NAOK=TRUE)
       bins=EV$bins
       bint=EV$bint
       lenbins=EV$lenbins
       lenbint=EV$lenbint
       lenbinst=EV$lenbinst
       moments=EV$moments
       momentt=EV$momentt
       momentst=EV$momentst
      centers <- bins[1:numvario]+diff(bins)/2
    #  if(type=="lorelogram"){
    #  elorel <- matrix(momentst,nrow=numvario,ncol=numbint,byrow=TRUE)
    #  elorel <- rbind(c(0,momentt),cbind(moments,elorel))
    #  a <- rowSums(elorel)==0
    #  b <- colSums(elorel)==0
    #  d <- rowSums(elorel)!=0
    #  f <- colSums(elorel)!=0
    #  if(sum(a)){
    #      elorel <- elorel[-which(a),]
    #      centers <- c(0,centers)[which(d)]
    #      centers <- centers[-1]}
    #  if(sum(b)){
    #      elorel <- elorel[,-which(b)]
    #      bint <- c(0,bint)[which(f)]
    #      bint <- bint[-1]}
    #  elorel[elorel==0] <- NA
    #  moments <- as.vector(elorel[,1][-1])
    #  momentt <- as.vector(elorel[1,][-1])
    #  momentst <- c(t(elorel[-1,-1]))
    #  lenbins <- rep(1,length(moments))
    #  lenbint <- rep(1,length(momentt))
    #  lenbinst <- rep(1,length(momentst))}
      indbin <- lenbins>0
      indbint <- lenbint>0
      indbinst <- lenbinst>0
      bins <- bins[indbin]
      bint <- bint[indbint]
      centers <- centers[indbin]
      moments <- moments[indbin]
      lenbins <- lenbins[indbin]
      momentt <- momentt[indbint]
      lenbint <- lenbint[indbint]
      momentst <- momentst[indbinst]
      lenbinst <- lenbinst[indbinst]
      variograms <- moments/lenbins
      variogramt <- momentt/lenbint
      variogramst <- momentst/lenbinst
    }
  #***********************************************************************************************#
  #***********************************************************************************************#
  #***********************************************************************************************#
    if(!initparam$bivariate&&!initparam$spacetime){  ## spatial univariate case
     fname <- paste(fname,"2",sep="") 
      if(grid)     {a=expand.grid(coordx,coordy);coordx=a[,1];coordy=a[,2]; }
     # Computes the spatial moments
      EV=.C(fname, bins=bins,  as.double(coordx),as.double(coordy),as.double(coordt),as.double(data), lenbins=lenbins,
         moments=moments, as.integer(numbins),PACKAGE='GeoModels', DUP = TRUE, NAOK=TRUE)
       bins=EV$bins
       lenbins=EV$lenbins
       moments=EV$moments
      # Computes the spatial variogram:
      indbin <- lenbins>0
      bins <- bins[indbin]
      numbins <- length(bins)
      # check if cloud or binned variogram:
      if(cloud) centers <- bins else centers <- bins[1:(numbins-1)]+diff(bins)/2
      moments <- moments[indbin]
      lenbins <- lenbins[indbin]
      variograms <- moments/lenbins}
    # Start --- compute the extremal coefficient
    .C('DeleteGlobalVar', PACKAGE='GeoModels', DUP = TRUE, NAOK=TRUE)
    EVariogram <- list(bins=bins,
                       bint=bint,
                       cloud=cloud,
                       centers=centers,
                       lenbins=lenbins,
                       lenbinst=lenbinst,
                       lenbint=lenbint,
                       maxdist =maxdist,
                       maxtime = maxtime,
                       variograms=variograms,
                       variogramst=variogramst,
                       variogramt=variogramt,
                       type=type)

    structure(c(EVariogram, call = call), class = c("GeoVariogram"))

  }


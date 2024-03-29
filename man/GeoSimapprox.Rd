\name{GeoSimapprox}
\alias{GeoSimapprox}
\encoding{UTF-8}
\title{Fast simulation of Gaussian and non Gaussian Random Fields.}
\description{
  Simulation of Gaussian and some non Gaussian spatial, spatio-temporal and spatial bivariate 
 random fields using approximate methods of simulation. At the time it works only for the spatial case and the Matern model
}
\usage{
GeoSimapprox(coordx, coordy=NULL, coordt=NULL, 
coordx_dyn=NULL,corrmodel, distance="Eucl",GPU=NULL, 
grid=FALSE,local=c(1,1),
method="TB",M=30, L=1000,model='Gaussian',
n=1,param,anisopars=NULL, radius=6371,X=NULL)
}
\arguments{
  \item{coordx}{A numeric (\eqn{d \times 2}{d x 2})-matrix (where
    \code{d} is the number of spatial sites) giving 2-dimensions of spatial coordinates or a numeric \eqn{d}{d}-dimensional vector giving
    1-dimension of spatial coordinates.
     Coordinates on a sphere for a  fixed radius \code{radius} 
    are passed in lon/lat format expressed in decimal degrees.}
  \item{coordy}{A numeric vector giving 1-dimension of
    spatial coordinates; \code{coordy} is interpreted only if \code{coordx} is a numeric
    vector or \code{grid=TRUE} otherwise it will be ignored. Optional argument, the default is \code{NULL} then \code{coordx} is expected to
    be numeric a (\eqn{d \times 2}{d x 2})-matrix.}
  \item{coordt}{A numeric vector giving 1-dimension of
    temporal coordinates.  Optional argument, the default is \code{NULL}
    then a spatial RF is expected.}
  \item{coordx_dyn}{A list of \eqn{m} numeric (\eqn{d_t \times 2}{d x 2})-matrices
       containing dynamical (in time) spatial coordinates. Optional argument, the default is \code{NULL}
    }
  \item{corrmodel}{String; the name of a correlation model, for the
    description see the Section \bold{Details}.}
  \item{distance}{String; the name of the spatial distance. The default
    is \code{Eucl}, the euclidean distance. See the Section
    \bold{Details}  of \code{\link{GeoFit}}.}
    \item{GPU}{Numeric; if \code{NULL} (the default) 
      no GPU computation is performed.} 

  \item{grid}{Logical; if \code{FALSE} (the default) the data
    are interpreted as spatial or spatial-temporal realisations on a set
    of non-equispaced spatial sites (irregular grid).}
          \item{local}{Numeric; number of local work-items of the GPU}
     \item{method}{String; the type of approximation method. Default is \code{TB}  that is the 
     turning band method. The other possible choice is \code{Vecchia} that is the Vechia method.} 
     \item{M}{Numeric; the number of neighboords in the Vecchia method.} 
      \item{L}{Numeric; the number of lines in the  turning band method.} 
  \item{model}{String; the type of RF and therefore the densities associated to the likelihood
    objects. \code{Gaussian} is the default, see the Section
    \bold{Details}.}
  \item{n}{Numeric; the number of trials for binomial RFs. The number of successes in the negative Binomial RFs. Default is \eqn{1}.} 
  \item{param}{A list of parameter values required in the
    simulation procedure of RFs, see \bold{Examples}.}
      \item{anisopars}{ A list of two elements "angle" and "ratio" i.e. the anisotropy angle and the anisotropy ratio, respectively.}
    \item{radius}{Numeric; a value indicating  the radius of the sphere when using the great 
    circle distance. Default value is the radius of the earth in Km (i.e. 6371)}  
  \item{X}{Numeric; Matrix of space-time covariates.}
}


\value{
  Returns an object of class \code{GeoSim}.
  An object of class \code{GeoSim} is a list containing
  at most the following components:
  
  \item{bivariate}{Logical:\code{TRUE} if the Gaussian RF is bivariate, otherwise  \code{FALSE};}
  \item{coordx}{A \eqn{d}{d}-dimensional vector of spatial coordinates;}
  \item{coordy}{A \eqn{d}{d}-dimensional vector of spatial coordinates;}
  \item{coordt}{A \eqn{t}{t}-dimensional vector of temporal coordinates;}
  \item{coordx_dyn}{A  list of dynamical (in time) spatial coordinates;}
  \item{corrmodel}{The correlation model; see \code{\link{GeoCovmatrix}}.}
  \item{data}{The vector or matrix or array of data, see
    \code{\link{GeoFit}};}
  \item{distance}{The type of spatial distance;}
  \item{method}{The method of simulation}
  \item{model}{The type of RF, see \code{\link{GeoFit}}.}
  \item{n}{The number of trial for Binomial RFs;the number of successes in a negative Binomial RFs;} 
  \item{numcoord}{The number of spatial coordinates;}
  \item{numtime}{The number the temporal realisations of the RF;}
  \item{param}{The vector of  parameters' estimates;}
  \item{radius}{The radius of the sphere if coordinates are passed in lon/lat format;}
  \item{randseed}{The seed used for the random simulation;}
  \item{spacetime}{\code{TRUE} if spatio-temporal and \code{FALSE} if
  spatial RF;}
}

\references{

J. Guiness (2018) Permutation and Grouping Methods for Sharpening Gaussian Process Approximations \emph{Technometrics} 60(4) 415-429.

D. Arroyo, X. Emery (2020) An R Implementation of a Continuous Spectral Algorithm for Simulating Vector Gaussian Random Fields in Euclidean Spaces ACM Transactions on Mathematical SoftwareVolume 47(1)

}

\author{Moreno Bevilacqua, \email{moreno.bevilacqua89@gmail.com},\url{https://sites.google.com/view/moreno-bevilacqua/home},
Víctor Morales Oñate, \email{victor.morales@uv.cl}, \url{https://sites.google.com/site/moralesonatevictor/},
Christian", Caamaño-Carrillo, \email{chcaaman@ubiobio.cl},\url{https://www.researchgate.net/profile/Christian-Caamano}
}


\examples{
library(GeoModels)

################################################################
###
### Example 1. Simulation of a large spatial Gaussian RF 
###            with Turning band method
###############################################################
set.seed(68)
x <- runif(50000)
y <- runif(50000)
coords=cbind(x,y)
# Simulation of a spatial Gaussian RF with Matern correlation function
data1 <- GeoSimapprox(coords, corrmodel="Matern", 
            method="TB",L=1000,
            param=list(smooth=0.5,mean=0,sill=1,scale=0.4/3,nugget=0))$data
quilt.plot(coords,data1)

}
\keyword{Simulation}

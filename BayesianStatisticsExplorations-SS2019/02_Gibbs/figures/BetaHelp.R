BetaShapeParms = function(
  m = 0.1, ##<< mean 
  s= 0.025 ##<< stdev
){
  #The mean is a/(a+b) and the variance is ab/((a+b)^2 (a+b+1)). 
  
  #Solve[{a/(a+b) ==m, ab/((a+b)^2 * (a+b+1)) == s2}, {a,b}]
  
  a = -(m*(m^2 - m + s^2))/s^2 
  b = ((m - 1)*(m^2 - m + s^2))/s^2
  return(c(a,b))
}

ggPlotBetaPrior = function(AlphaBeta= cbind(alpha=c(0.5,1,5,1,2,2),beta=c(0.5,1,1,3,2,5)),
                            ylim = c(0,4), xlim = c(0,1), log ="", lwd=1, leg="top",
                           lty, col, 
                           xlab="",ylab="prior", legSuffix="", NormalApprox = FALSE){
  M = nrow(AlphaBeta)
  
  x = seq(0.001,1,length=500)
  i=1;xy= cbind.data.frame(x,y=dbeta(x,AlphaBeta[i,1],AlphaBeta[i,2]), ab= paste(AlphaBeta[i,1],AlphaBeta[i,2],sep=", "))
  for (i in 2:M) {
    xyi= cbind.data.frame(x,y=dbeta(x,AlphaBeta[i,1],AlphaBeta[i,2]), ab= paste(AlphaBeta[i,1],AlphaBeta[i,2],sep=", "))
    xy = rbind.data.frame(xy,  xyi)
  }
  if(NormalApprox){
    for (i in 1:M) {
      NN = sum(AlphaBeta[i,])
      p = AlphaBeta[i,1]/NN
      x=seq(p-0.05,p+0.05,length=501)
      s=sqrt(p*(1-p)/NN)
      xyi= cbind.data.frame(x,y=dnorm(x,p,s), ab= paste0("p=",p))
      xy = rbind.data.frame(xy,  xyi)
    }
  }
  # for (p in c(0.1,0.2)){
  #   x=seq(p-0.05,p+0.05,length=501)
  #   s=sqrt(p*(1-p)/5000)
  #   #lines(x,dnorm(x,p,s), col = "green", lty = 3)
  #   
  # }
  
  p = ggplot(xy, aes(x=x,y=y)) + geom_line(aes(lty=ab,colour=ab), size=lwd)
   # labs(colour = expression(alpha~","~beta), x=xlab,y=ylab) +
  #if (missing())
  p = p +  labs(linetype = expression(alpha~","~beta), colour= expression(alpha~","~beta), x=xlab,y=ylab)
    scale_linetype(guide = 'none')
  
  if (grepl("x", log)) p = p + scale_x_log10(limits = xlim) else p = p + scale_x_continuous(limits = xlim)
  if (grepl("y", log)) p = p + scale_y_log10(limits = ylim) else p = p + scale_y_continuous(limits = ylim)
  
  return(p)
}

PlotBetaPrior = function(AlphaBeta= cbind(alpha=c(0.5,1,5,1,2,2),beta=c(0.5,1,1,3,2,5)),
                         ylim = c(0,4), xlim = NULL, log ="", leg="top", legText,
                         legSuffix="", col = 1:nrow(AlphaBeta)){
  M = nrow(AlphaBeta)
  y= matrix(0,500,ncol=M)
  x = seq(0.001,1,length=500)
  for (i in 1:M) y[,i] = dbeta(x,AlphaBeta[i,1],AlphaBeta[i,2])
  matplot(x,y,type="l", ylim = ylim , xlim = xlim, lty=1:M, lwd=2, log=log, xlab="p", col=col);
  grid()
  if (!is.null(leg)){
    if (missing(legText)) {
      param_strings = paste0( AlphaBeta[,"alpha"], ", ", AlphaBeta[,"beta"], legSuffix)
      legTitle=expression(alpha~","~beta)  
    } else {
      param_strings=legText
      legTitle=NULL
    }
    legend(leg, legend = param_strings, col=1:M, lty=1:M, title=legTitle);
  }
  
  #if (RanGen) return()
  invisible(list(x,y))
}

if (0){
  ab = BetaShapeParms(0.1, 0.025)
  PlotBetaPrior(cbind(alpha=ab[1],beta=ab[2]),xlim = c(0.1+c(-3,3)*0.025),ylim=NULL)
}
## This code is used to test the KCCU method

args <- commandArgs(TRUE)
fname = args[1]
output = args[2]


library(kernlab)
library(corpcor)
library(MCMCpack)

######################################
Fish<-function(r){
  (1/2)*log((1+r)/(1-r))
}
######################################
mat.center<-function(mat){
 m.c<-sweep(mat,2,apply(mat,2,mean),"-") 
 m.c.r<-sweep(m.c,1,apply(m.c,1,mean),"-")
 return(m.c.r)
}
######################################
kern.out<-function(dat,per){  
   
   #Sigma estimation
   sig<-sigest(dat,frac=1,scale=F)[[2]]
   rbf<-rbfdot(sigma=sig)
   k.mat<-kernelMatrix(rbf,dat)

   #Centering
   k.mat<-mat.center(k.mat)

   #Regularization
   repeat{
    k.svd<-svd(k.mat,nu=0)
    k.lead<-sum(cumsum(k.svd$d^2/sum(k.svd$d^2))<per)
    k.lead.ed<-max(c(2,k.lead))
    if(is.na(k.lead.ed)==F){
      break
    }
   }
   return(k.mat%*%k.svd$v[,1:k.lead.ed])
}
   
kern.out.big<-function(dat,rand,per){
  #dat: large N data
  #rand: random number of kernel basis functions to use
  #per: total percentage of explained variance to use
  
  #Sigma estimation
  sig<-sigest(dat,frac=1)[[2]]
  rbf<-rbfdot(sigma=sig)
  k.mat<-kernelMatrix(rbf,scale(dat))

  #Centering
  k.mat<-mat.center(k.mat)

  #Regularization
  k.mat<-k.mat[,sample(1:ncol(k.mat),rand)]
  k.svd<-fast.svd(k.mat)
  k.lead<-sum(cumsum(k.svd$d^2/sum(k.svd$d^2))<per)
  k.lead.ed<-max(c(2,k.lead))
  k.lead.ed<-min(c(k.lead.ed,25))
  return(k.mat%*%k.svd$v[,1:k.lead.ed])
}
######################################
trimmed.jack<-function(T.jack,a){
  n<-length(T.jack)
  which.a<-quantile(T.jack,c(a,1-a))
  T.trim<-T.jack[which(T.jack>=which.a[1]&T.jack<=which.a[2])]
  r.a<-length(T.trim)
  T.a<-mean(T.trim)
  s2.a<-(1/n)*(1/(1-2*a)^2)*((1/n)*sum((T.trim-T.a)^2)+a*(min(T.trim)-T.a)^2+a*(max(T.trim)-T.a)^2)
  return(list(T.a=T.a,var.a=s2.a))
}
######################################
gene.kcca<-function(k1.case,k2.case,k1.cont,k2.cont,trim){
  #data.case/data.cont:  Nxm format of data
  #trim: trimming level for trimmed-jackknife
  n.case<-nrow(k1.case)
  n.cont<-nrow(k1.cont)
  
  kcca.case<-cancor(k1.case,k2.case)
  kcca.cont<-cancor(k1.cont,k2.cont)
  
  kcca.case.x<-kcca.case$xcoef
  kcca.case.y<-kcca.case$ycoef
  kcca.cont.x<-kcca.cont$xcoef
  kcca.cont.y<-kcca.cont$ycoef
  
  r.case<-kcca.case$cor[1]
  r.cont<-kcca.cont$cor[1]
  
  z.case<-Fish(r.case)
 	z.cont<-Fish(r.cont)

  cc.case.jack<-rep(NA,n.case) 
  for(j in 1:n.case){
      cc.case<-cancor(k1.case[-j,],k2.case[-j,])
      cc.case.xrot<-procrustes(cc.case$xcoef,kcca.case.x)$X.new
      cc.case.yrot<-procrustes(cc.case$ycoef,kcca.case.y)$X.new
      cc.case.jack[j]<-max(Fish(cor(k1.case[-j,]%*%cc.case.xrot[,1,drop=F],k2.case[-j,]%*%cc.case.yrot[,1,drop=F])),0)
  }
  theta.case.jack<-n.case*z.case-(n.case-1)*cc.case.jack
  mean.case<-mean(theta.case.jack)  
  var.case<-(1/(n.case*(n.case-1)))*sum((theta.case.jack-mean.case)^2)
  
  cc.cont.jack<-rep(NA,n.cont)      
  for(j in 1:n.cont){
      cc.cont<-cancor(k1.cont[-j,],k2.cont[-j,])
      cc.cont.xrot<-procrustes(cc.cont$xcoef,kcca.cont.x)$X.new
      cc.cont.yrot<-procrustes(cc.cont$ycoef,kcca.cont.y)$X.new
      cc.cont.jack[j]<-max(Fish(cor(k1.cont[-j,]%*%cc.cont.xrot[,1,drop=F],k2.cont[-j,]%*%cc.cont.yrot[,1,drop=F])),0)
  }
  theta.cont.jack<-n.cont*z.cont-(n.cont-1)*cc.cont.jack
  mean.cont<-mean(theta.cont.jack)
  var.cont<-(1/(n.cont*(n.cont-1)))*sum((theta.cont.jack-mean.cont)^2)
  
  trim.case<-trimmed.jack(theta.case.jack,trim)
  trim.cont<-trimmed.jack(theta.cont.jack,trim)
  
  Z.case.trim<-max(trim.case[[1]],0)
  Z.cont.trim<-max(trim.cont[[1]],0)
  var.jack<-var.case+var.cont
  var.trim<-trim.case[[2]]+trim.cont[[2]]
  Z.stat<-(Z.case.trim-Z.cont.trim)/sqrt(var.trim)
  p.val<-2*(1-pnorm(abs(Z.stat),0,1))
  return(list(z.case=Z.case.trim,z.cont=Z.cont.trim,var.trim=var.trim,Z.stat=Z.stat,p.val=p.val,trim=trim))
}

#######################################


#data <- read.table("0.005_0.2_0.2_0.05_data_EDM-02_1-1000-000",header=F)
data <- read.table(file=fname,header=F)
res <- list()

for(i in 1:4){
  for(j in (i+1):5){
    g1.case <- as.matrix(data[which(data['V51']==0),((i-1)*10+1):((i-1)*10+10)])
    g1.cont <- as.matrix(data[which(data['V51']==1),((i-1)*10+1):((i-1)*10+10)])
    g2.case <- as.matrix(data[which(data['V51']==0),((j-1)*10+1):((j-1)*10+10)])
    g2.cont <- as.matrix(data[which(data['V51']==1),((j-1)*10+1):((j-1)*10+10)]) 
    k1.case <- kern.out(g1.case,0.9)
    k1.cont <- kern.out(g1.cont,0.9)
    k2.case <- kern.out(g2.case,0.9)
    k2.cont <- kern.out(g2.cont,0.9)
    

    res <- gene.kcca(k1.case,k2.case,k1.cont,k2.cont,0.05)
    phr <- sprintf("%d %d %f", i, j, res$p.val)
    write(phr,file=output,append=TRUE)
  }
}


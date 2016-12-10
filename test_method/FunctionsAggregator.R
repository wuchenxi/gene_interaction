require(mvtnorm)

##source("https://bioconductor.org/biocLite.R")
##biocLite("snpStats")

require(snpStats)

EM.iter <- function(M,f=c(1/4,1/4,1/4,1/4))
{
	## Expectation
	PTilde <- rep(0,times=10)
	PTilde[1] <- f[1]*f[1]
	PTilde[2] <- 2*f[1]*f[2]
	PTilde[3] <- 2*f[1]*f[3]
	PTilde[4] <- 2*f[1]*f[4]
	PTilde[5] <- f[2]*f[2]
	PTilde[6] <- 2*f[2]*f[3]
	PTilde[7] <- 2*f[2]*f[4]
	PTilde[8] <- f[3]*f[3]
	PTilde[9] <- 2*f[3]*f[4]
	PTilde[10] <- f[4]*f[4]

	## Maximization
	PPheno <- matrix(0,ncol=3,nrow=3)
	PPheno[1,1] <- f[1]*f[1] ## AABB
	PPheno[1,2] <- 2*f[1]*f[2] ## AABb
	PPheno[1,3] <- f[2]*f[2] ## AAbb
	PPheno[2,1] <- 2*f[1]*f[3] ## AaBB
	PPheno[2,2] <- 2*f[1]*f[4]+2*f[2]*f[3] ## AaBb
	PPheno[2,3] <- 2*f[2]*f[4] ## Aabb
	PPheno[3,1] <- f[3]*f[3] ## aaBB
	PPheno[3,2] <- 2*f[3]*f[4] ## aaBb
	PPheno[3,3] <- f[4]*f[4] ## aabb
	
	P <- rep(0,times=10) ## 1.AB_AB 2.AB_Ab ... 5. Ab_Ab ... 8. aB_aB .. 10. ab_ab
	P[1] <- (M[1,1]/(sum(M)))*PTilde[1]/PPheno[1,1] ## AB_AB
	P[2] <- (M[1,2]/(sum(M)))*PTilde[2]/PPheno[1,2] ## AB_Ab
	P[3] <- (M[2,1]/(sum(M)))*PTilde[3]/PPheno[2,1] ## AB_aB
	P[4] <- (M[2,2]/(sum(M)))*PTilde[4]/PPheno[2,2] ## AB_ab
	P[5] <- (M[1,3]/(sum(M)))*PTilde[5]/PPheno[1,3] ## Ab_Ab
	P[6] <- (M[2,2]/(sum(M)))*PTilde[6]/PPheno[2,2] ## Ab_aB
	P[7] <- (M[2,3]/(sum(M)))*PTilde[7]/PPheno[2,3] ## Ab_ab
	P[8] <- (M[3,1]/(sum(M)))*PTilde[8]/PPheno[3,1] ## aB_aB
	P[9] <- (M[3,2]/(sum(M)))*PTilde[9]/PPheno[3,2] ## aB_ab
	P[10] <- (M[3,3]/(sum(M)))*PTilde[10]/PPheno[3,3] ## ab_ab

	Newf <- rep(0,times=4)
	Newf[1] <- (1/2)*(2*P[1]+P[2]+P[3]+P[4])
	Newf[2] <- (1/2)*(P[2]+2*P[5]+P[6]+P[7])
	Newf[3] <- (1/2)*(P[3]+P[6]+2*P[8]+P[9])
	Newf[4] <- (1/2)*(P[4]+P[7]+P[9]+2*P[10])
	return(Newf)
}

get.freq <- function(M,nmax=100,tol=1e-10){ ### M matrice 3x3
	f0 <- EM.iter(M)
	f1 <- EM.iter(M,f0)
	iter <- 1
	while ( (iter < nmax) && (sum((f1-f0)^2,na.rm=TRUE) > tol )){
		f0 <- f1
		f1 <- EM.iter(M,f0)
		iter <- iter+1
	}
	if (iter==nmax){print(paste("Optimisation reaches maximum iteration:",nmax))}
	return(f1)
}

getM <- function(R,S1,S2){
	M <- matrix(NA,ncol=3,nrow=3)
	M[1,1] <- length(which(R[,S1]==0 & R[,S2]==0))
	M[1,2] <- length(which(R[,S1]==0 & R[,S2]==1))
	M[1,3] <- length(which(R[,S1]==0 & R[,S2]==2))
	M[2,1] <- length(which(R[,S1]==1 & R[,S2]==0))
	M[2,2] <- length(which(R[,S1]==1 & R[,S2]==1))
	M[2,3] <- length(which(R[,S1]==1 & R[,S2]==2))
	M[3,1] <- length(which(R[,S1]==2 & R[,S2]==0))
	M[3,2] <- length(which(R[,S1]==2 & R[,S2]==1))
	M[3,3] <- length(which(R[,S1]==2 & R[,S2]==2))
	return(M)
}

get.r <- function(R,S1,S2){
	M <- getM(R,S1,S2)
	my.f <- get.freq(M)
	p <- my.f[4]+my.f[3]
	q <- my.f[4]+my.f[2]
	pq <- my.f[4]
	return((pq-p*q)/sqrt(p*(1-p)*q*(1-q)))
}

get.MatCor.1Region <- function(X){
	res <- matrix(NA,ncol=ncol(X),nrow=ncol(X))
	if (ncol(X)==1){
		res[1,1] <- 1
	} else{
		for (i in 1:(ncol(X)-1)){
			for (j in (i+1):ncol(X)){
				res[i,j] <- res[j,i] <- get.r(X,i,j)
			}
		}
		diag(res) <- 1
	}
	return(res)
}

get.MatCor.2Regions <- function(X1,X2){
	MatCor1 <- 	get.MatCor.1Region(X1)
	MatCor2 <- 	get.MatCor.1Region(X2)
	n1 <- ncol(X1)
	n2 <- ncol(X2)
	n.pairs <- n1*n2
	res <- matrix(NA,ncol=n.pairs,nrow=n.pairs)
	for (i in 1:(n.pairs-1)){
		i1 <- floor((i-1)/n2)+1
		j1 <- i-(i1-1)*n2
		for (j in (i+1):n.pairs){
			i2 <- floor((j-1)/n2)+1
			j2 <- j-(i2-1)*n2
			res[i,j] <- res[j,i] <- MatCor1[i1,i2]*MatCor2[j1,j2]
		}
	}
	diag(res) <- 1
	return(res)
}

minP.test=function(p.val,corMat,method="Conneely",sample.minP=NULL,n.sample=1000){
	if (method=="Conneely"){
		minP = min(p.val)
		n.test = ncol(corMat)
		return(1-pmvnorm(lower=-rep(qnorm(1-minP/2),n.test),upper=rep(qnorm(1-minP/2),n.test),corr=corMat,abseps = 0.01))
		}
	if (method=="Sampling"){
		if (is.null(sample.minP)){
			sample.T <- rmvnorm(n.sample,sigma=corMat)
			sample.p.val <- 2*(1-pnorm(sample.T))
			sample.minP <- apply(sample.p.val,1,min)
		}
		minP = min(p.val)
		return(mean(sample.minP < minP))
	}
}


Aggregator.internal <- function(Y,X1,X2){
	if (is.matrix(X1)){X1 <- as.data.frame(X1)}
	if (is.matrix(X2)){X2 <- as.data.frame(X2)}
	if (!is.data.frame(X1)){stop("X1 is not an object of class matrix or data.frame")}
	if (!is.data.frame(X2)){stop("X2 is not an object of class matrix or data.frame")}
	if ( (length(Y)!=nrow(X1)) || (length(Y)!=nrow(X2)) || (nrow(X1)!=nrow(X2))){
		stop("Y, X1 and X2 must have the same length")
	}
	if (any(Y != 0 & Y !=1)){stop("Y (Phenotype) must only have values in (0,1)")}
	if (any(X1 != 0 & X1 !=1 & X1 !=2)){stop("X1 (Genotype of the first gene) must only have values in (0,1)")}
	if (any(X2 != 0 & X2 !=1 & X2 !=2)){stop("X1 (Genotype of the first gene) must only have values in (0,1)")}
	MatCor <- get.MatCor.2Regions(X1,X2)
	MatCor[is.nan(MatCor)] <- median(MatCor[which(!is.na(MatCor))])
	MatCor[is.na(MatCor)] <- median(MatCor[which(!is.na(MatCor))])
	n1 <- ncol(X1)
	n2 <- ncol(X2)
	pval.vec <- rep(NA,times=n1*n2)
	compt <- 1
	for (i in 1:n1){
		for (j in 1:n2){
			pval.vec[compt] <- anova(glm(Y ~ X1[,i]*X2[,j],family="binomial"),test="Chisq")[4,5]
			compt <- compt+1
		}
	}
	pval.vec[is.na(pval.vec)] <- median(pval.vec[which(!is.na(pval.vec))])
	pval.vec[is.nan(pval.vec)] <- median(pval.vec[which(!is.nan(pval.vec))])
	tmp <- minP.test(pval.vec,corMat=MatCor)
	if (attributes(tmp)$msg=="Completion with error > abseps"){
		warning(paste("p-values are approximated with error=",attributes(tmp)$error))
	}
	return(list(p.value=tmp[1]))
}

import.PED <- function(ped.file,info.file){
	sample <- read.pedfile(file=ped.file,snps=info.file)
	Y <- sample$fam$affected-1
	X <- as(sample$genotype,"numeric")
	return(list(Y=Y,X=X))
}

import.plink <- function(bed.file,bim.file,fam.file){
	sample <- read.plink(bed=bed.file,bim=bim.file,fam=fam.file)
	Y <- sample$fam$affected-1
	X <- as(sample$genotypes,"numeric")
	return(list(Y=Y,X=X))
}

Aggregator <- function(Y=NULL,X1=NULL,X2=NULL,ped.file=NULL,info.file=NULL,bed.file=NULL,bim.file=NULL,fam.file=NULL,snp.R1=NULL,snp.R2=NULL,type="Normal"){
	if (type=="Normal"){
		if (is.null(Y)){stop("Y (Phenotype) is missing")}
		if (is.null(X1)){stop("X1 (Genotype of the first gene) is missing")}
		if (is.null(X2)){stop("X2 (Genotype of the second gene) is missing")}
		return(Aggregator.internal(Y,X1,X2))
	}
	if (type=="PED"){
		if (is.null(ped.file)){stop("Argument bed.file is missing")}
		if (is.null(info.file)){stop("Argument info.file is missing")}
		sample <- import.PED(ped.file=ped.file,info.file=info.file)
		X1 <- sample$X[,snp.R1]
		X2 <- sample$X[,snp.R2]
		Y <- sample$Y
		return(Aggregator.internal(Y,X1,X2))
	}
	if (type=="Plink"){
		if (is.null(bed.file)){stop("Argument bed.file is missing")}
		if (is.null(bim.file)){stop("Argument bim.file is missing")}
		if (is.null(fam.file)){stop("Argument fam.file is missing")}
		sample <- import.plink(bed=bed.file,bim=bim.file,fam.file=fam.file)
		X1 <- sample$X[,snp.R1]
		X2 <- sample$X[,snp.R2]
		Y <- sample$Y
		return(Aggregator.internal(Y,X1,X2))
	}
}

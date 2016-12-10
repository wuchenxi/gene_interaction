# This code is used to test emily's method

args <- commandArgs(TRUE)
sourcepath = args[1]
fname = args[2]
output = args[3]

#path <- "/home/ruby/Documents/emily changed/"

#source(paste(path,"FunctionsAggregator.R",sep=""))
source(sourcepath)

#data <-as.matrix(read.table("/home/ruby/Documents/emily changed/0.025_44_1_1000_EDM-3_0001.txt",header=T,sep=""))
geno <-as.matrix(read.table(file=fname,header=F,sep=""))
res <- list()

for( i in 1:4){
  for (j in (i+1):5){
    X1 <- geno[,((i-1)*10+1):((i-1)*10+10)]
    X2 <- geno[,((j-1)*10+1):((j-1)*10+10)]
    Y <- geno[,51]
    res <- Aggregator(Y,X1,X2)
    phr <- sprintf("%d %d %f",i,j,res)
    write(phr, file=output,append = TRUE)
  }
}

#X1<-M[,1:100]
#X2<-M[,101:200]
#Y<-M[,1001]

## Normal format
#Aggregator(Y,X1,X2)

## PED format
##Aggregator(ped.file=paste(path,"CA1_PADI4.ped",sep=""),info.file=paste(path,"CA1_PADI4.info",sep=""),snp.R1=1:15,snp.R2=16:19,type="PED")


## Plink format
##Aggregator(bed.file=paste(path,"CA1_PADI4.bed",sep=""),bim.file=paste(path,"CA1_PADI4.bim",sep=""),fam.file=paste(path,"CA1_PADI4.fam",sep=""),snp.R1=1:15,snp.R2=16:19,type="Plink")

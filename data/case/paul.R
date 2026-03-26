#https://www.bioconductor.org/packages/release/bioc/vignettes/tradeSeq/inst/doc/tradeSeq.html

library(tradeSeq)
library(slingshot)

path <- "~/BA/data/case/"

data(countMatrix, package = "tradeSeq")
counts <- as.matrix(countMatrix)
data(crv, package = "tradeSeq")

saveRDS(counts, file=paste0(path, "paulDataset.rds"))
saveRDS(crv, file=paste0(path, "paulCrv.rds"))
#https://www.bioconductor.org/packages/release/bioc/vignettes/tradeSeq/inst/doc/tradeSeq.html

library(tradeSeq)

path <- "~/BA/data/paul/sim_1/"

if (!dir.exists(path)) dir.create(path, recursive = TRUE)

data(countMatrix, package = "tradeSeq")
counts <- as.matrix(countMatrix)
data(crv, package = "tradeSeq")

saveRDS(counts, file=paste0(path, "dataset.rds"))
saveRDS(crv, file=paste0(path, "crv.rds"))
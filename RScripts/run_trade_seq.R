# https://www.bioconductor.org/packages/release/bioc/vignettes/tradeSeq/inst/doc/tradeSeq.html

library(tradeSeq)
library(RColorBrewer)
library(slingshot)

FQnorm <- function(counts) {
  rk <- apply(counts, 2, rank, ties.method = 'min')
  counts.sort <- apply(counts, 2, sort)
  refdist <- apply(counts.sort, 1, median)
  norm <- apply(rk, 2, function(r) { refdist[r] })
  rownames(norm) <- rownames(counts)
  return(norm)
}

dataset <- readRDS(paste0("~/BA/data/paul/sim_1/dataset.rds"))
crv <- readRDS(paste0("~/BA/data/paul/sim_1/crv.rds"))

#counts <- t(as.matrix(dataset$counts))
counts <- FQnorm(dataset)
pseudotime <- slingPseudotime(crv, na = FALSE)
cellWeights <- slingCurveWeights(crv)

# For a proper evaluation k should be chosen using evaluateK()
sce <- fitGAM(counts = counts, pseudotime = pseudotime, cellWeights = cellWeights, nknots = 6, verbose = FALSE)
table(rowData(sce)$tradeSeq$converged)

patternRes <- startVsEndTest(sce)
head(patternRes[order(patternRes$waldStat, decreasing = TRUE), ])
print(top_gene)
print(which(rownames(counts) == top_gene))
plotSmoothers(sce, counts, gene = top_gene)

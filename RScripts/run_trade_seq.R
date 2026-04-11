# https://www.bioconductor.org/packages/release/bioc/vignettes/tradeSeq/inst/doc/tradeSeq.html

library(tradeSeq)
library(RColorBrewer)
library(slingshot)

dataset <- readRDS(paste0("~/BA/data/bifurcating/sim_1/dataset.rds"))
crv <- readRDS(paste0("~/BA/data/bifurcating/sim_1/crv.rds"))

counts <- t(as.matrix(dataset$counts))
pseudotime <- slingPseudotime(crv, na = FALSE)
cellWeights <- slingCurveWeights(crv)

# For a proper evaluation k should be chosen using evaluateK()
sce <- fitGAM(counts = counts, pseudotime = pseudotime, cellWeights = cellWeights, nknots = 6, verbose = FALSE)
table(rowData(sce)$tradeSeq$converged)

#patternRes <- startVsEndTest(sce)
#head(patternRes[order(patternRes$waldStat, decreasing = TRUE), ])
#top_gene <- rownames(patternRes)[order(patternRes$waldStat, decreasing = TRUE)[1]]
plotSmoothers(sce, counts, gene = 13)

# https://www.bioconductor.org/packages/release/bioc/vignettes/tradeSeq/inst/doc/tradeSeq.html

library(tradeSeq)
library(RColorBrewer)

data <- readRDS(paste0("~/BA/data/simulated/bifurcating/datasets/bifurcatingDyntoyDataset_", 1, ".rds"))
crv <- readRDS(paste0("~/BA/data/simulated/bifurcating/crv/bifurcatingSlingshotCrv_", 1, ".rds"))

counts <- t(data$counts)
pseudotime <- slingPseudotime(crv, na = FALSE)
cellWeights <- slingCurveWeights(crv)

# For a proper evaluation k should be chosen using evaluateK()
sce <- fitGAM(counts = counts, pseudotime = pseudotime, cellWeights = cellWeights, nknots = 6, verbose = FALSE)
table(rowData(sce)$tradeSeq$converged)

patternRes <- startVsEndTest(sce)
head(patternRes[order(patternRes$waldStat, decreasing = TRUE), ])
top_gene <- rownames(patternRes)[order(patternRes$waldStat, decreasing = TRUE)[1]]
plotSmoothers(sce, counts, gene = top_gene)

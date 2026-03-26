# https://www.bioconductor.org/packages/devel/bioc/vignettes/slingshot/inst/doc/vignette.html#using-slingshot
# /tradeSeqPaper/simulation/sim2_dyntoy_bifurcating_420190326_evaluateAllDyntoyBifurcating_R3.6.R

library(mclust)
library(RColorBrewer)

FQnorm <- function(counts){
  rk <- apply(counts,2,rank,ties.method='min')
  counts.sort <- apply(counts,2,sort)
  refdist <- apply(counts.sort,1,median)
  norm <- apply(rk,2,function(r){ refdist[r] })
  rownames(norm) <- rownames(counts)
  return(norm)
}

for(datasetIter in 1:10){
  
  # Loading the data set
  data <- readRDS(paste0("~/BA/data/simulated/bifurcating/datasets/bifurcatingDyntoyDataset_", datasetIter, ".rds"))
  counts <- t(data$counts)
  
  # Gene Filtering (not done in the tradeSeq paper)
  # geneFilter <- rowSums(counts >= 3) >= 10
  # counts_filtered <- counts[geneFilter, ]
  # tde_filtered <- data$tde_overall[geneFilter, ]
  
  # Normalization
  counts_norm <- FQnorm(counts)
  
  # Dimensionality Reduction
  pca <- prcomp(log1p(t(counts_norm)), scale. = FALSE)
  rd <- pca$x[,1:3]

  # Clusters (hardcoded in the tradeSeq paper)
  cl <- Mclust(rd)$classification
  
  # Identify start and end clusters (hardcoded in the tradeSeq paper)
  start_node <- cl[which.min(data$prior_information$timecourse_continuous)]
  
  ends <- data$prior_information$end_milestones
  milestones <- data$prior_information$groups_id$group_id
  time <- data$prior_information$timecourse_continuous
  
  end_node_1 <- cl[which.max(time * (milestones == ends[1]))]
  end_node_2 <- cl[which.max(time * (milestones == ends[2]))]
  
  # Lineages
  lin <- getLineages(rd, cl, start.clus = start_node, end.clus = c(end_node_1, end_node_2))
  
  # Curves
  crv <- getCurves(lin)
  
  plot(rd, col = brewer.pal(9,"Set1")[cl], asp = 1, pch = 16)
  lines(SlingshotDataSet(crv), lwd = 3, col = 'black')
  
  saveRDS(crv, file=paste0("~/BA/data/simulated/bifurcating/crv/bifurcatingSlingshotCrv_",datasetIter,".rds"))
}
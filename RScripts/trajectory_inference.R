# https://www.bioconductor.org/packages/devel/bioc/vignettes/slingshot/inst/doc/vignette.html#using-slingshot
# /tradeSeqPaper/simulation/sim2_dyntoy_bifurcating_420190326_evaluateAllDyntoyBifurcating_R3.6.R
# /tradeSeqPaper/simulation/sim2_dyngen_cycle_72/20190403_evaluateCyclicAll_includingEdgeR.R
library(princurve)
library(mclust)
library(RColorBrewer)
library(wesanderson)
library(Matrix)
library(slingshot)

path_base <- "~/BA/data/"

FQnorm <- function(counts) {
  rk <- apply(counts,2,rank,ties.method='min')
  counts.sort <- apply(counts,2,sort)
  refdist <- apply(counts.sort,1,median)
  norm <- apply(rk,2,function(r){ refdist[r] })
  rownames(norm) <- rownames(counts)
  return(norm)
}

ti_furcating <- function(data) {
  counts <- t(as.matrix(data$counts))
  
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
  end_nodes <- c()
  
  for (i in 1:length(ends)) {
    end_node <- cl[which.max(time * (milestones == ends[i]))]
    end_nodes <- c(end_nodes, end_node)
  }
  
  # Lineages
  lin <- getLineages(rd, cl, start.clus = start_node, end.clus = end_nodes)
  
  # Curves
  crv <- getCurves(lin)
  
  # Plot
  plot(rd, col = brewer.pal(9,"Set1")[cl], asp = 1, pch = 16)
  lines(SlingshotDataSet(crv), lwd = 3, col = 'black')
  
  return(crv)
}

ti_cyclic <- function(data) {
  counts <- t(as.matrix(data$counts))
  
  # Normalization
  counts_norm <- FQnorm(counts)
  
  # Dimensionality Reduction
  pca <- prcomp(log1p(t(counts_norm)), scale. = FALSE)
  rd <- pca$x[,1:2] # tradeSeq used pca$x[,1:2] for the cyclic data only
  
  # Curves
  pcc <- principal_curve(rd, smoother="periodic_lowess")
  
  # Plot
  plot(rd, col = 'grey', asp = 1, pch = 16)
  lines(x=pcc$s[order(pcc$lambda),1], y=pcc$s[order(pcc$lambda),2], col="red", lwd=2)
  
  return(pcc)
}

# Bifurcating
set.seed(1)
for (i in 1:10) {
  sim_dir <- file.path(path_base, "bifurcating", paste0("sim_", i))
  data <- readRDS(file.path(sim_dir, "dataset.rds"))
  crv <- ti_furcating(data)
  saveRDS(crv, file = file.path(sim_dir, "crv.rds"))
}

# Multifrucating
set.seed(1)
for (i in 1:10) {
  sim_dir <- file.path(path_base, "multifurcating", paste0("sim_", i))
  data <- readRDS(file.path(sim_dir, "dataset.rds"))
  crv <- ti_furcating(data)
  saveRDS(crv, file = file.path(sim_dir, "crv.rds"))
}

# Cyclic
set.seed(1)
for (i in 1:10) {
  sim_dir <- file.path(path_base, "cyclic", paste0("sim_", i))
  data <- readRDS(file.path(sim_dir, "dataset.rds"))
  crv <- ti_cyclic(data)
  saveRDS(crv, file = file.path(sim_dir, "crv.rds"))
}
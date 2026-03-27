library(tradeSeq)
library(slingshot)

path_base <- "~/BA/data/"

FQnorm <- function(counts) {
  rk <- apply(counts, 2, rank, ties.method = 'min')
  counts.sort <- apply(counts, 2, sort)
  refdist <- apply(counts.sort, 1, median)
  norm <- apply(rk, 2, function(r) { refdist[r] })
  rownames(norm) <- rownames(counts)
  return(norm)
}


extract_trajectory_data <- function(data, crv) {
  # Paul data has to be treated special
  counts <- if ("counts" %in% names(data)) data$counts else data
  
  # Normalize counts
  counts <- t(as.matrix(counts))
  counts <- FQnorm(counts)
  counts <- t(counts) # Output: Cells x Genes
  
  # The pcc curve from the cyclic data has to be treated special
  if (inherits(crv, "SlingshotDataSet")) {
    pseudotime  <- slingPseudotime(crv, na = FALSE)
    weights     <- slingCurveWeights(crv)
  } else {
    pseudotime  <- crv$lambda
    weights     <- rep(1, length(pseudotime)) # Because it has only 1 lineage
  }
  
  return(list(counts = counts, pseudotime = pseudotime, weights = weights))
}


process_dataset <- function(category, iter = NULL) {
  # Paul dataset does not have multiple simulations
  if (is.null(iter)) {
    instance_dir <- paste0(path_base, category, "/")
  } else {
    instance_dir <- paste0(path_base, category, "/sim_", iter, "/")
  }
  
  data_path <- paste0(instance_dir, "dataset.rds")
  crv_path  <- paste0(instance_dir, "crv.rds")
  
  data <- readRDS(data_path)
  crv  <- readRDS(crv_path)
  res  <- extract_trajectory_data(data, crv)
  
  write.csv(res$counts,     file = paste0(instance_dir, "counts.csv"))
  write.csv(res$pseudotime, file = paste0(instance_dir, "pseudotime.csv"))
  write.csv(res$weights,    file = paste0(instance_dir, "weights.csv"))
}

categories <- c("bifurcating", "multifurcating", "cyclic")

for (cat in categories) {
  for (i in 1:10) {
    process_dataset(cat, i)
  }
}

process_dataset("paul")
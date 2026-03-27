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

extract_furcating <- function(data, crv) {
  counts <-  data$counts
  
  # Normalize counts
  counts <- t(as.matrix(counts))
  counts <- FQnorm(counts)
  counts <- t(counts) # Output: Cells x Genes
  
  pseudotime  <- slingPseudotime(crv, na = FALSE)
  weights     <- slingCurveWeights(crv)
  tde         <- data$tde_overall
  
  return(list(counts = counts, pseudotime = pseudotime, weights = weights, tde = tde))
}

extract_cyclic <- function(data, crv) {
  counts <-  data$counts
  
  # Normalize counts
  counts <- t(as.matrix(counts))
  counts <- FQnorm(counts)
  counts <- t(counts) # Output: Cells x Genes
  
  pseudotime  <- crv$lambda
  weights     <- rep(1, length(pseudotime)) # Because it has only 1 lineage
  
  tde <- data.frame(
    feature_id = data$feature_info$feature_id,
    differentially_expressed = !data$feature_info$is_hk # Anything that is not housekeeping is DE
  )
  
  return(list(counts = counts, pseudotime = pseudotime, weights = weights, tde = tde))
}

extract_paul <- function(data, crv) {
  counts <- data
  
  # Normalize counts
  counts <- t(as.matrix(counts))
  counts <- FQnorm(counts)
  counts <- t(counts) # Output: Cells x Genes
  
  pseudotime  <- slingPseudotime(crv, na = FALSE)
  weights     <- slingCurveWeights(crv)
  
  # No tde data available for the Paul dataset since it is a case study
  tde <- data.frame(
    feature_id = colnames(counts),
    differentially_expressed = NA 
  )
  
  return(list(counts = counts, pseudotime = pseudotime, weights = weights, tde = tde))
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
  
  if (category == "bifurcating" || category == "multifurcating") {
    res  <- extract_furcating(data, crv)
  } else if (category == "cyclic") {
    res  <- extract_cyclic(data, crv)
  } else if (category == "paul") {
    res  <- extract_paul(data, crv)
  }
  
  write.csv(res$counts,     file = paste0(instance_dir, "counts.csv"))
  write.csv(res$pseudotime, file = paste0(instance_dir, "pseudotime.csv"))
  write.csv(res$weights,    file = paste0(instance_dir, "weights.csv"))
  write.csv(res$tde,        file = paste0(instance_dir, "tde.csv"), row.names = FALSE)
}

categories <- c("bifurcating", "multifurcating", "cyclic")

for (cat in categories) {
  for (i in 1:10) {
    process_dataset(cat, i)
  }
}

process_dataset("paul")
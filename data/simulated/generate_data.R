library(tidyverse)
library(dyno)
library(dyntoy)
library(dynutils)
library(dyngen)
library(dynplot)
library(patchwork)

save_path <- "~/BA/data/simulated/datasets/"
  
# Bifurcating data sets
# tradeSeqPaper/simulation/sim2_dyntoy_bifurcating_4/20190326_generate_dyntoy_dataset.R

# set.seed(1)
# 
# for (i in 1:10) {
#   dataset <- dyntoy::generate_dataset(
#     model = model_bifurcating(),
#     num_cells = 500,
#     num_features = 5000,
#     differentially_expressed_rate = .2
#   )
# 
#   saveRDS(dataset, file=paste0(save_path, "bifurcating/bifurcatingDataset_", i, ".rds"))
# }

# Multifurcating data sets
# /tradeSeqPaper/simulation/sim2_dyntoy_multifurcating_4/20190326_evaluateDyntoyMultifurcating4.Rmd

# set.seed(1)
# 
# for (i in 1:10) {
#   dataset <- dyntoy::generate_dataset(
#     model = model_multifurcating(),
#     num_cells = 750,
#     num_features = 5000,
#     differentially_expressed_rate = .2
#   )
# 
#   saveRDS(dataset, file=paste0(save_path, "multifurcating/multifurcatingDataset_", i, ".rds"))
# }

# Cyclic data sets
# /tradeSeqPaper/simulation/sim2_dyngen_cycle_72/simulate.R
# I could not find the exact generation script so I'm going off the values in the paper
# https://dyngen.dynverse.org/

set.seed(1)

backbone <- backbone_cycle()

config <- initialise_model(
  backbone = backbone,
  num_tfs = nrow(backbone$module_info),
  num_cells = 500,
  num_targets = 180,    # 400 genes x 45 % DE genes
  num_hks = 220,        # 400 genes x 55 % non-DE genes
  verbose = TRUE
)

for (i in 1:10) {
  out <- dyngen::generate_dataset(
    config,
    format = "dyno",
    make_plots = FALSE
  )
  
  saveRDS(out$dataset, file=paste0(save_path, "cyclic/cyclicDataset_", i, ".rds"))
}

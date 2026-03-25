# tradeSeqPaper-master/simulation/sim2_dyntoy_bifurcating_4/20190326_generate_dyntoy_dataset.R

library(tidyverse)
library(dyno)
library(dyntoy)
library(patchwork)


set.seed(1)
for (i in 1:10) {
  dataset <- generate_dataset(
    model = model_bifurcating(),
    num_cells = 500,
    num_features = 5000,
    differentially_expressed_rate = .2
  )
  
  saveRDS(dataset, file=paste0("~/BA/data/simulated/bifurcating/datasets/bifurcatingDyntoyDataset_", i, ".rds"))
}
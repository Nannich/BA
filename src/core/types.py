from dataclasses import dataclass
import numpy as np

@dataclass
class SingleCellDataset:
    raw_counts: np.ndarray          # Shape: (n_cells, n_genes) -> float32
    log_counts: np.ndarray          # Shape: (n_cells, n_genes) -> float32
    pseudotime: np.ndarray          # Shape: (n_cells, n_lineages) -> float32
    lineage_assignment: np.ndarray  # Shape: (n_cells, n_lineages) -> bool
    gene_names: np.ndarray          # Shape: (n_genes,) -> str
    cell_names: np.ndarray          # Shape: (n_cells,) -> str

    def slice_cells(self, mask):
        """Safely slices all cell-axis structures in lockstep."""
        return SingleCellDataset(
            raw_counts=self.raw_counts[mask, :],
            log_counts=self.log_counts[mask, :],
            pseudotime=self.pseudotime[mask, :],
            lineage_assignment=self.lineage_assignment[mask, :],
            gene_names=self.gene_names,
            cell_names=self.cell_names[mask]
        )
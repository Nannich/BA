def resolve_gene_identifier(gene_input, dataset):
    """
    Translates an input identifier (int, numeric string, or name string) 
    into a validated dataset integer column index.
    """
    if gene_input is None:
        return None

    gene_names_list = list(dataset.gene_names)

    if isinstance(gene_input, int):
        gene_idx = gene_input
    elif str(gene_input).isdigit():
        gene_idx = int(gene_input)
    else:
        if gene_input in gene_names_list:
            gene_idx = gene_names_list.index(gene_input)
        else:
            raise ValueError(
                f"Gene '{gene_input}' could not be located in this dataset.\n"
            )

    n_genes = dataset.log_counts.shape[1]
    if gene_idx >= n_genes or gene_idx < 0:
        raise IndexError(
            f"Resolved gene index {gene_idx} is out of bounds for this dataset containing {n_genes} genes."
        )

    return gene_idx
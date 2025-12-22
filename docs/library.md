# Library

## By DIA-NN (> v1.9, recommended) 
  1. Generate the predicted library with the .predicted.speclib suffix based on sequence databases in UniProt format (see [instructions](https://github.com/vdemichev/DiaNN#getting-started-with-dia-nn)).
  2. Convert the .predicted.speclib file to .parquet format (see [instructions](https://github.com/vdemichev/DiaNN?tab=readme-ov-file#editing-spectral-libraries)).

## By others

In this way, the .tsv or .parquet library should contain these columns:

* **Precursor.Id** - peptide seq + precursor charge.
* **Modified.Sequence** - peptide seq with modifications, only supporting C(UniMod:4) and M(UniMod:35).
* **Stripped.Sequence** - peptide seq.
* **Precursor.Charge** - the charge of the precursor.
* **Proteotypic** - whether the peptide is proteotypic (i.e., uniquely mapping to a single protein).
* **Decoy** - 0. Full-DIA will generate the decoys itself.
* **N.Term** - N-terminal enzymatic specificity of the peptide.
* **C.Term** - C-terminal enzymatic specificity of the peptide.
* **RT** - retention time or iRT or predicted RT of the peptide.
* **IM** - ion mobility or predicted ion mobility of the precursor.
* **Q.Value** - 0.
* **Peptidoform.Q.Value** - 0.
* **PTM.Site.Confidence** - 0.
* **PG.Q.Value** - 0.
* **Precursor.Mz** - m/z of the precursor.
* **Product.Mz** - m/z of the fragment ion.
* **Relative.Intensity** - relative intensity of the fragment ion.
* **Fragment.Type** - "b" or "y".
* **Fragment.Charge** - 1 or 2.
* **Fragment.Series.Number** - the number of aas of the fragment ion.
* **Fragment.Loss.Type** - "noloss"
* **Exclude.From.Quant** - 0.
* **Protein.Ids** - all UniProt Ids of proteins matched to the peptides in the library.
* **Protein.Group** - None.
* **Protein.Names** - all UniProt names of proteins matched to the peptides in the library.

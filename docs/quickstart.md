# Quickstart

```bash
full_dia -lib "Absolute path of the spectral library" -ws "Absolute path of the .d folder or a folder containing multiple .d folders"
```
(Please note that the path needs to be enclosed in quotes if running on a Windows platform.)

- `-lib`<br>
This parameter is used to specify the absolute path of the spectral library.
Full-DIA currently supports spectral libraries with the ***.parquet*** or ***.tsv*** suffix, provided that their column names are consistent with those of the DIA-NN (> v1.9) predicted spectral library. 
We recommend generating the predicted spectral library using DIA-NN and then converting it to the .parquet format.
Refer to [this](https://github.com/vdemichev/DiaNN) for instructions on how to generate prediction spectral libraries and convert to .parquet format using DIA-NN.
Full-DIA supports oxygen modifications on methionine (M) but does not include modifications such as phosphorylation or acetylation.
Full-DIA will develop its own predictor capable of forecasting the peptide retention time, ion mobility, and fragmentation pattern. 
It may also be compatible with other formats of spectral libraries based on requests.

- `-ws`<br>
This parameter specifies the folder that contains multiple .d directories to be analyzed.

Other optional params are list below by entering `full_dia -h`:
```
       ******************
       * Full-DIA x.y.z *
       ******************
Usage: full_dia -ws WS -lib LIB

optional arguments for users:
  -h, --help           Show this help message and exit.
  -ws WS               Specify the folder that is .d or contains .d files.
  -lib LIB             Specify the absolute path of a .speclib or .parquet spectra library.
  -out_name OUT_NAME   Specify the folder name of outputs. Default: full_dia.
  -gpu_id GPU_ID       Specify the GPU-ID (e.g. 0, 1, 2) which will be used. Default: 0.
```

### Output
Full-DIA will generate **`report.log.txt`** and **`report.parquet`** in output folder. 
The report.parquet contains precursor and protein IDs, as well as plenty of associated information. 
Most column names are consistent with DIA-NN and are self-explanatory.

* **Protein.Group** - inferred proteins. Full-DIA uses [IDPicker](https://pubs.acs.org/doi/abs/10.1021/pr070230d) algorithm to infer proteins. 
* **Protein.Ids** - all proteins matched to the precursor in the library.
* **Protein.Names** - names (UniProt names) of the proteins in the Protein.Group.
* **PG.Quantity.Raw** - raw quantity of the Protein.Group.
* **PG.Quantity.Deep** - corrected quantity of the Protein.Group.
* **Precursor.Id** - peptide seq + precursor charge.
* **Precursor.Charge** - the charge of precursor.
* **Q.Value** - run-specific precursor q-value.
* **Global.Q.Value** - global precursor q-value.
* **PG.Q.Value** - run-specific q-value for the protein group.
* **Global.PG.Q.Value** - global q-value for the protein group.
* **Proteotypic** - indicates the peptide is specific to a protein.
* **Precursor.Quantity.Raw** - raw quantity of the precursor.
* **Precursor.Quantity.Deep** - corrected quantity of the precursor.
* **RT** - the retention time of the precursor.
* **IM** - the ion mobility of the precursor.

from pathlib import Path

import numpy as np
import pandas as pd

from full_dia.log import Logger

try:
    _ = profile
except NameError:

    def profile(func):
        return func


logger = Logger.get_logger()


class Library:
    """
    Reader class of the spectral library.
    """

    def __init__(self, dir_lib: str):
        """
        Load the library.
        """
        dir_lib = Path(dir_lib)
        logger.info("Loading lib: " + dir_lib.name)
        self.lib_type = dir_lib.suffix

        # parquet or tsv
        if self.lib_type == ".parquet":
            df = pd.read_parquet(dir_lib)
        elif self.lib_type == ".tsv":
            df = pd.read_csv(dir_lib, sep="\t")
        else:
            raise ValueError(
                f"Unsupported spectral library format: '{self.lib_type}'. "
                "Only .parquet and .tsv are supported."
            )
        self.check_lib(df)

        df = df[df["Decoy"] == 0].reset_index(drop=True)
        self.df_pr, self.df_map = self.construct_dfs(df)

        assert len(self.df_pr) == self.df_pr["pr_id"].nunique()
        logger.info(f"Lib prs: {len(self.df_pr)}")

    def check_lib(self, df: pd.DataFrame) -> None:
        """
        Check spectral library:
            column names, modifications, charges, loss, proteins, length
        """
        required_columns = {
            "Precursor.Id",
            "Modified.Sequence",
            "Stripped.Sequence",
            "Precursor.Charge",
            "Proteotypic",
            "Decoy",
            "N.Term",
            "C.Term",
            "RT",
            "IM",
            "Q.Value",
            "Peptidoform.Q.Value",
            "PTM.Site.Confidence",
            "PG.Q.Value",
            "Precursor.Mz",
            "Product.Mz",
            "Relative.Intensity",
            "Fragment.Type",
            "Fragment.Charge",
            "Fragment.Series.Number",
            "Fragment.Loss.Type",
            "Exclude.From.Quant",
            "Protein.Ids",
            "Protein.Group",
            "Protein.Names",
        }
        # check name
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"The spectral library is missing required columns: {sorted(missing_cols)}. "
            )

        # check modification
        x = df["Modified.Sequence"].copy()
        x = x.drop_duplicates()
        x = x.replace([r"C\(UniMod:4\)", r"M\(UniMod:35\)"], ["c", "m"], regex=True)
        if x.str.contains(r"[\(\)0-9]").any():
            raise ValueError(
                "The spectral library contains unexpected modifications. "
                "Only C(UniMod:4) and M(UniMod:35) are allowed."
            )

        # check charge
        x = df["Precursor.Charge"].max()
        if x > 4:
            raise ValueError(
                "The spectral library contains > 4 charge state. "
                "Only charge 1-4 are allowed."
            )

        # check protein
        x = df["Protein.Ids"].str.count(";") != df["Protein.Names"].str.count(";")
        if x.any():
            raise ValueError(
                "The spectral library contains inconsistent Protein.IDs and Protein.Names."
            )

        # check fg type
        x1 = df["Fragment.Type"].isin(["b", "y"]).all()
        x2 = (df["Fragment.Loss.Type"] == "noloss").all()
        if not (x1 and x2):
            raise ValueError(
                "The spectral library can only contain b/y fragment ions without neutral losses."
            )

        # check fg length. Full-DIA will encode 'b12_1' to 1121
        if df["Fragment.Series.Number"].max() >= 100:
            raise ValueError(
                "The spectral library can not contain b/y fragment ions with >= 100 aas."
            )

    def __len__(self):
        return len(self.df_pr)

    @profile
    def construct_dfs(self, df: pd.DataFrame) -> tuple:
        """
        Construct the df_pr and df_map from DIA-NN's .parquet library.

        Parameters
        ----------
        df : pd.DataFrame
            The raw DIA-NN's .parquet file.

        Returns
        -------
        tuple
            df_pr : pd.DataFrame
                Each row corresponds to a precursor and its fragment information.

            df_map : pd.DataFrame
                Each row represents the protein information corresponding to the peptide in the same row of df_pr.
        """
        # info - pr
        fg_height_v = df["Relative.Intensity"].values.astype(np.float32)

        fg_height_max_idx = np.where(fg_height_v == fg_height_v.max())[0]
        pr_id_v = df.loc[fg_height_max_idx, "Precursor.Id"]
        good_idx = ~pr_id_v.duplicated()  # for case: 1., 1., 0.8
        fg_height_max_idx = fg_height_max_idx[good_idx]

        pr_id_v = df.loc[fg_height_max_idx, "Precursor.Id"].values
        pr_charge_v = df.loc[fg_height_max_idx, "Precursor.Charge"].values
        pr_mz_v = df.loc[fg_height_max_idx, "Precursor.Mz"].values
        pred_irt_v = df.loc[fg_height_max_idx, "RT"].values
        pred_iim_v = df.loc[fg_height_max_idx, "IM"].values
        pr_length_v = df.loc[fg_height_max_idx, "Stripped.Sequence"].str.len().values

        # info - fg
        fg_num_v = np.diff(fg_height_max_idx)
        fg_num_v = np.append(fg_num_v, len(df) - fg_height_max_idx[-1])
        fg_mz_v = df["Product.Mz"].values.astype(np.float32)
        fg_type_v = np.where(df["Fragment.Type"] == "b", 1, 2)
        fg_type_v = fg_type_v.astype(np.int16)  # b-1, y-2
        fg_index_v = df["Fragment.Series.Number"].values.astype(np.int16)
        fg_charge_v = df["Fragment.Charge"].values.astype(np.int16)
        fg_anno_v = fg_type_v * 1000 + fg_index_v * 10 + fg_charge_v
        mask = np.arange(fg_num_v.max()) < fg_num_v[:, None]
        fg_mz = np.zeros(mask.shape, dtype=np.float32)
        fg_mz[mask] = fg_mz_v
        fg_height = np.zeros(mask.shape, dtype=np.float32)
        fg_height[mask] = fg_height_v
        fg_anno = np.ones(mask.shape, dtype=np.int16) * 3011  # 3 is fg_loss
        fg_anno[mask] = fg_anno_v

        # df_map
        protein_id_v = df.loc[fg_height_max_idx, "Protein.Ids"].values
        protein_name_v = df.loc[fg_height_max_idx, "Protein.Names"].values
        # pg_v = df.loc[fg_height_max_idx, 'Protein.Group'].values
        gene_v = df.loc[fg_height_max_idx, "Genes"].values
        df_map = pd.DataFrame()
        df_map["protein_id"] = protein_id_v
        df_map["protein_name"] = protein_name_v
        df_map["gene"] = gene_v
        x = df_map["protein_id"].str.count(";")
        y = df_map["protein_name"].str.count(";")
        assert (x == y).all(), "Protein ID/Name are not corresponding relationships!"

        # df_pr
        df_pr = pd.DataFrame()
        df_pr["pr_id"] = pr_id_v
        df_pr["pr_charge"] = pr_charge_v.astype(np.int8)
        df_pr["pr_mz"] = pr_mz_v.astype(np.float32)
        df_pr["pr_len"] = pr_length_v.astype(np.int8)
        df_pr["pred_irt"] = pred_irt_v.astype(np.float32)
        df_pr["pred_iim"] = pred_iim_v.astype(np.float32)
        df_pr["fg_num"] = fg_num_v.astype(np.int8)
        df_pr["pr_index"] = df_pr.index.values.astype(np.int32)
        fg_num = fg_mz.shape[1]
        df_pr[["fg_mz_" + str(i) for i in range(fg_num)]] = fg_mz
        df_pr[["fg_height_" + str(i) for i in range(fg_num)]] = fg_height
        df_pr[["fg_anno_" + str(i) for i in range(fg_num)]] = fg_anno

        assert len(df_pr) == len(df_map)

        return df_pr, df_map

    # @profile
    def polish_lib_by_swath(
        self, swath: np.ndarray, ws_diann: Path | None = None
    ) -> pd.DataFrame:
        """
        Remove prs whose m/z values are not in the range of SWATH settings.

        Parameters
        ----------
        swath : np.ndarray
            The SWATH settings.

        ws_diann : Path, default=None
            For developing.

        Returns
        -------
        df_lib : pd.DataFrame
            The polished library.
        """
        df_lib = self.df_pr

        # for developing
        if ws_diann is not None:
            df_diann = pd.read_csv(ws_diann / "diann" / "report.tsv", sep="\t")
            df_diann = df_diann[df_diann["Q.Value"] < 0.01]
            df_diann = df_diann[
                [
                    "Modified.Sequence",
                    "Precursor.Charge",
                    "RT",
                    "IM",
                    "Precursor.Quantity",
                ]
            ]
            df_diann["diann_rt"] = df_diann["RT"] * 60.0
            df_diann["diann_im"] = df_diann["IM"]
            df_diann["diann_pr_quant"] = df_diann["Precursor.Quantity"]
            df_diann["pr_id"] = df_diann["Modified.Sequence"] + df_diann[
                "Precursor.Charge"
            ].astype(str)
            df = pd.merge(df_lib, df_diann, on="pr_id")
            df = df.reset_index(drop=True)
            del df["Modified.Sequence"]
            del df["Precursor.Charge"]
            del df["RT"]
            del df["IM"]
            del df["Precursor.Quantity"]
            df_lib = df

        # screen prs by range of m/z
        pr_mz = df_lib["pr_mz"].values
        pr_mz_min, pr_mz_max = swath[0], swath[-1]
        good_idx = (pr_mz > pr_mz_min) & (pr_mz < pr_mz_max)
        df_lib = df_lib.iloc[good_idx].reset_index(drop=True)

        # drop duplicates
        df_lib = df_lib.drop_duplicates(subset="pr_id", ignore_index=True)
        assert len(df_lib) == df_lib.pr_id.nunique()

        # remove BJOUXZ
        df_lib["simple_seq"] = (
            df_lib["pr_id"]
            .str[:-1]
            .replace(["C\(UniMod:4\)", "M\(UniMod:35\)"], ["c", "m"], regex=True)
        )
        bad_idx = df_lib["simple_seq"].str.contains("[BJOUXZ]", regex=True)
        df_lib = df_lib[~bad_idx].reset_index(drop=True)

        # fg_num >= 4
        df_lib = df_lib[df_lib.fg_num >= 4]
        df_lib = df_lib.reset_index(drop=True)

        # pred_im
        df_lib["pred_im"] = df_lib["pred_iim"]

        # pr_mz_iso
        mass_neutron = 1.0033548378
        pr_mass = df_lib["pr_mz"] * df_lib["pr_charge"]
        pr_mz_1H = (pr_mass + mass_neutron) / df_lib["pr_charge"]
        pr_mz_2H = (pr_mass + 2 * mass_neutron) / df_lib["pr_charge"]
        pr_mz_left = (pr_mass - mass_neutron) / df_lib["pr_charge"]
        df_lib["pr_mz_1H"] = pr_mz_1H.astype(np.float32)
        df_lib["pr_mz_2H"] = pr_mz_2H.astype(np.float32)
        df_lib["pr_mz_left"] = pr_mz_left.astype(np.float32)

        # assign swath_id
        swath_id = np.digitize(df_lib["pr_mz"].values, swath)
        df_lib["swath_id"] = swath_id.astype(np.int8)
        idx = np.argsort(swath_id)
        df_lib = df_lib.iloc[idx].reset_index(drop=True)

        # decoy
        df_lib["decoy"] = np.uint8(0)

        # shuffle
        np.random.seed(1)
        df_lib = df_lib.sample(frac=1, random_state=1).reset_index(drop=True)

        logger.info(f"Polishing spectral library: {len(df_lib)}prs")

        return df_lib

    def assign_proteins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign proteins based on the precursor index from raw df_map.

        Parameters
        ----------
        df : pd.DataFrame
            Provide the "pr_index" column.

        Returns
        -------
        df : pd.DataFrame
            Add new columns: "protein_id", "protein_name", "proteotypic"
        """
        # find corresponding protein and name by pr_index
        if self.lib_type == ".parquet":
            df_map = self.df_map
            pr_index_q = df["pr_index"].values
            result_protein_id = df_map.loc[pr_index_q, "protein_id"]
            result_protein_name = df_map.loc[pr_index_q, "protein_name"]

        df["protein_id"] = result_protein_id.values
        df["protein_name"] = result_protein_name.values

        df["proteotypic"] = df["protein_id"].str.count(";") + 1
        df.loc[df["proteotypic"] != 1, "proteotypic"] = 0

        # add DECOY
        if "decoy" in df.columns:
            decoy_idx = df["decoy"] == 1
            df.loc[decoy_idx, "protein_id"] = "DECOY_" + df.loc[decoy_idx, "protein_id"]
            df.loc[decoy_idx, "protein_id"] = df.loc[decoy_idx, "protein_id"].replace(
                ";", ";DECOY_", regex=True
            )

            df.loc[decoy_idx, "protein_name"] = (
                "DECOY_" + df.loc[decoy_idx, "protein_name"]
            )
            df.loc[decoy_idx, "protein_name"] = df.loc[
                decoy_idx, "protein_name"
            ].replace(";", ";DECOY_", regex=True)

        return df

    def assign_fg_mz(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign fg mz values based on the precursor index from raw df_pr.

        Parameters
        ----------
        df : pd.DataFrame
            Provide the "pr_index" column.

        Returns
        -------
        df : pd.DataFrame
            Add the m/z value columns of fragment ions.
        """
        cols = ["fg_mz_" + str(i) for i in range(12)]
        pr_index_q = df["pr_index"].values
        df[cols] = self.df_pr.loc[pr_index_q, cols].values
        return df

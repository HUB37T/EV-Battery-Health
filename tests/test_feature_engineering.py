"""Feature arithmetic, missing/unknown handling, and frozen-base checks."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import EngineeredPreprocessor, pipeline_state_fingerprint
from sklearn.preprocessing import StandardScaler
from src.preprocessing import BatteryPreprocessor


class FeatureEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame({
            "cycle_count":[10.,20.,30.,40.,50.],"battery_age_days":[5.,10.,15.,20.,25.],
            "total_energy_throughput_kwh":[100.,220.,330.,440.,550.],
            "avg_ambient_temp_c":[20.,21.,22.,23.,24.],"peak_temp_during_fast_charge_c":[30.,32.,33.,34.,35.],
            "fast_charge_ratio":[.5,.2,.3,.4,.6],"avg_depth_of_discharge_pct":[80.,70.,60.,50.,40.],
            "recall_flag":[0.,0.,1.,0.,0.],"chemistry":["LFP","NMC","LFP","NMC","LFP"],
        })
        self.base=BatteryPreprocessor(
            numeric_features=[c for c in self.frame if c not in {"chemistry","recall_flag"}],
            categorical_features=["chemistry"],binary_features=["recall_flag"],
        ).fit(self.frame)

    def make(self,blocks=("usage_rates","context","chemistry")):
        return EngineeredPreprocessor(self.base,blocks).fit(self.frame)

    def test_formula_values_and_units(self):
        prep=self.make();raw=prep._arithmetic(self.frame)
        self.assertEqual(raw.loc[0,"fe_cycles_per_day"],2.)
        self.assertEqual(raw.loc[0,"fe_energy_per_cycle_kwh"],10.)
        self.assertEqual(raw.loc[0,"fe_charge_ambient_delta_c"],10.)
        self.assertEqual(raw.loc[0,"fe_fastcharge_dod_interaction"],.4)
        self.assertAlmostEqual(raw.loc[0,"fe_chem_LFP_log_cycles"],np.log1p(10))
        self.assertEqual(raw.loc[0,"fe_chem_NMC_log_cycles"],0.)

    def test_nonpositive_denominators_and_missing_operands_stay_missing(self):
        prep=self.make();future=self.frame.iloc[:3].copy()
        future.loc[0,"battery_age_days"]=0.
        future.loc[1,"cycle_count"]=0.
        future.loc[2,"total_energy_throughput_kwh"]=np.nan
        raw=prep._arithmetic(future)
        self.assertTrue(pd.isna(raw.loc[0,"fe_cycles_per_day"]))
        self.assertTrue(pd.isna(raw.loc[1,"fe_energy_per_cycle_kwh"]))
        self.assertTrue(pd.isna(raw.loc[2,"fe_energy_per_cycle_kwh"]))
        self.assertTrue(np.isfinite(prep.transform(future)).all())
        names=list(prep.get_feature_names_out())
        self.assertEqual(prep.transform(future)[0,names.index("missing__fe_cycles_per_day")],1.)

    def test_unknown_or_missing_chemistry_is_not_a_fake_known_chemistry(self):
        prep=self.make();future=self.frame.iloc[:2].copy()
        future.loc[0,"chemistry"]="UNSEEN"
        future.loc[1,"chemistry"]=None
        columns=[c for c in prep.extra_columns_ if c.startswith("fe_chem_")]
        self.assertTrue(prep._arithmetic(future)[columns].isna().all().all())
        self.assertTrue(np.isfinite(prep.transform(future)).all())

    def test_first_base_columns_and_base_state_are_preserved_exactly(self):
        before=joblib.hash(self.base);prep=self.make()
        expected=self.base.transform(self.frame)
        actual=prep.transform(self.frame)
        np.testing.assert_array_equal(actual[:,:expected.shape[1]],expected)
        self.assertEqual(before,joblib.hash(self.base))
        self.assertEqual(prep.base_state_hash_,joblib.hash(prep.base_preprocessor))

    def test_extra_fit_is_train_only_and_validation_transform_does_not_refit(self):
        prep=self.make();before=joblib.hash(prep)
        future=self.frame.iloc[:1].copy();future["total_energy_throughput_kwh"]=1e8
        prep.transform(future)
        self.assertEqual(before,joblib.hash(prep))
        self.assertAlmostEqual(prep.medians_["fe_energy_per_cycle_kwh"],11.)

    def test_target_or_id_cannot_be_smuggled_into_raw_inputs(self):
        prep=self.make()
        with self.assertRaises(ValueError):prep.transform(self.frame.assign(state_of_health_pct=90.))
        with self.assertRaises(ValueError):prep.transform(self.frame.assign(battery_id="b1"))

    def test_serialization_and_row_independence(self):
        prep=self.make();expected=prep.transform(self.frame)
        target=StandardScaler().fit(np.arange(len(self.frame)).reshape(-1,1))
        fingerprint=pipeline_state_fingerprint(prep,target)
        permuted=self.frame.iloc[::-1]
        np.testing.assert_array_equal(prep.transform(permuted),expected[::-1])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"feature_pipeline.joblib";joblib.dump((prep,target),path)
            loaded,loaded_target=joblib.load(path)
            np.testing.assert_array_equal(loaded.transform(self.frame),expected)
            self.assertEqual(pipeline_state_fingerprint(loaded,loaded_target),fingerprint)

    def test_fingerprint_detects_changed_learned_scaling(self):
        prep=self.make();target=StandardScaler().fit(np.arange(len(self.frame)).reshape(-1,1))
        before=pipeline_state_fingerprint(prep,target)
        prep.scaler_.mean_[0]+=1
        self.assertNotEqual(pipeline_state_fingerprint(prep,target),before)

    def test_unsupported_or_duplicate_blocks_are_rejected(self):
        with self.assertRaises(ValueError):EngineeredPreprocessor(self.base,["target_ratio"])
        with self.assertRaises(ValueError):EngineeredPreprocessor(self.base,["context","context"])


if __name__=="__main__":unittest.main()

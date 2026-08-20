"""Config tests: version presets, validation, receptive field, cache horizons."""

import warnings
import unittest

from engrama.config import VERSION_PRESETS, EngramaConfig


class TestVersionPresets(unittest.TestCase):
    def test_v3_preset_resolution(self):
        cfg = EngramaConfig(version="v3")
        self.assertEqual(cfg.synapse_mode, "factorized")
        self.assertEqual(cfg.cell_mode, "shared_core")
        self.assertEqual(cfg.offset_mode, "hierarchical_dyadic")
        self.assertEqual(cfg.cache_mode, "hierarchical")
        self.assertEqual(cfg.evoker_mode, "factorized")
        self.assertTrue(cfg.identity_transport)
        self.assertTrue(cfg.hierarchical_gate)

    def test_v2_preset_resolution(self):
        cfg = EngramaConfig(version="v2")
        self.assertEqual(cfg.synapse_mode, "dense")
        self.assertEqual(cfg.cell_mode, "independent")
        self.assertEqual(cfg.offset_mode, "dense_dilated")
        self.assertEqual(cfg.cache_mode, "full")
        self.assertEqual(cfg.evoker_mode, "dense")
        self.assertFalse(cfg.identity_transport)
        self.assertFalse(cfg.hierarchical_gate)

    def test_explicit_override_wins_over_preset(self):
        # Ablation of V3 spec 54: V2 base with factorized synapses only.
        cfg = EngramaConfig(version="v2", synapse_mode="factorized")
        self.assertEqual(cfg.synapse_mode, "factorized")
        self.assertEqual(cfg.offset_mode, "dense_dilated")  # stays V2
        self.assertEqual(cfg.cache_mode, "full")

    def test_v1_maps_to_dense_parameterization(self):
        cfg = EngramaConfig(version="v1")
        self.assertEqual(cfg.synapse_mode, VERSION_PRESETS["v1"]["synapse_mode"])


class TestValidation(unittest.TestCase):
    def test_invalid_values_raise(self):
        with self.assertRaises(ValueError):
            EngramaConfig(d_model=32, d_gate=32)  # d_gate must be < d_model
        with self.assertRaises(ValueError):
            EngramaConfig(num_candidates=0)
        with self.assertRaises(ValueError):
            EngramaConfig(num_candidates=9)
        with self.assertRaises(ValueError):
            EngramaConfig(candidate_aggregation="softmax")
        with self.assertRaises(ValueError):
            EngramaConfig(version="v4")
        with self.assertRaises(ValueError):
            EngramaConfig(offset_mode="spiral")
        with self.assertRaises(ValueError):
            EngramaConfig(cache_mode="elastic")
        with self.assertRaises(ValueError):
            EngramaConfig(offsets=[0, -1])
        with self.assertRaises(ValueError):
            EngramaConfig(d_model=32, synapse_rank=64)  # r must be <= d

    def test_depth_rule_warning(self):
        # L=3 with N=256 -> reach 7 << 255 -> warning of spec section 26
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            EngramaConfig(
                num_consolidation_layers=3, context_length=256,
                offset_mode="hierarchical_dyadic",
            )
        self.assertTrue(any("Depth rule" in str(wi.message) for wi in w))

    def test_no_warning_when_coverage_ok(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            EngramaConfig(num_consolidation_layers=8, context_length=256)
        self.assertFalse(any("Depth rule" in str(wi.message) for wi in w))


class TestOffsetsAndReceptiveField(unittest.TestCase):
    def test_hierarchical_dyadic_offsets(self):
        cfg = EngramaConfig(
            num_consolidation_layers=5, context_length=64,
            offset_mode="hierarchical_dyadic",
        )
        self.assertEqual(
            cfg.layer_offsets(),
            [[0, 1], [0, 1, 2], [0, 1, 4], [0, 1, 8], [0, 1, 16]],
        )

    def test_binary_minimal_offsets(self):
        cfg = EngramaConfig(
            num_consolidation_layers=4, context_length=64, offset_mode="binary_minimal"
        )
        self.assertEqual(cfg.layer_offsets(), [[0, 1], [0, 2], [0, 4], [0, 8]])

    def test_dense_dilated_uses_explicit_offsets(self):
        cfg = EngramaConfig(
            num_consolidation_layers=2, context_length=64,
            offset_mode="dense_dilated", offsets=[0, 1, 4],
        )
        self.assertEqual(cfg.layer_offsets(), [[0, 1, 4], [0, 1, 4]])

    def test_global_anchor_only_on_last_layer(self):
        cfg = EngramaConfig(
            num_consolidation_layers=3, context_length=16, global_anchor=True
        )
        offsets = cfg.layer_offsets()
        self.assertNotIn(15, offsets[0])
        self.assertNotIn(15, offsets[1])
        self.assertIn(15, offsets[2])

    def test_receptive_field_binary(self):
        cfg = EngramaConfig(num_consolidation_layers=8, context_length=256)
        rf = cfg.receptive_field()
        self.assertEqual(rf["max_reach"], 255)
        self.assertTrue(rf["dense_coverage"])
        self.assertTrue(rf["covers_context"])
        self.assertEqual(rf["required_layers_for_full_coverage"], 8)

    def test_cache_horizons(self):
        cfg = EngramaConfig(num_consolidation_layers=4, context_length=64)
        # D = [0,1],[0,1,2],[0,1,4],[0,1,8] -> capacities = max(D_{l+1})+1; last=1
        self.assertEqual(cfg.cache_horizons(), [3, 5, 9, 1])


class TestConfigSerialization(unittest.TestCase):
    def test_dict_round_trip(self):
        cfg = EngramaConfig(d_model=64, d_gate=8, num_candidates=2)
        cfg2 = EngramaConfig.from_dict(cfg.to_dict())
        self.assertEqual(cfg.to_dict(), cfg2.to_dict())

    def test_ignores_unknown_keys(self):
        cfg = EngramaConfig.from_dict({"d_model": 64, "obsolete_field": True})
        self.assertEqual(cfg.d_model, 64)


class TestSizePresets(unittest.TestCase):
    def test_all_presets_construct(self):
        for size in ("tiny", "small", "base", "large"):
            cfg = EngramaConfig.preset(size)
            self.assertEqual(cfg.version, "v3")
            rf = cfg.receptive_field()
            self.assertTrue(
                rf["covers_context"],
                f"preset {size} does not cover its context: {rf}",
            )


if __name__ == "__main__":
    unittest.main()

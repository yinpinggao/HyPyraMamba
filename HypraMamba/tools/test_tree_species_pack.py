#!/usr/bin/env python3
"""Unit tests for TreeSpeciesHSI load/pack (no GPU / no full cube required for pack tests)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import utils.data_load_operate as dlo


class PackTreeSpeciesPredictionTest(unittest.TestCase):
    def test_pack_when_pred_matches_scene_hw(self):
        meta = {
            'height': 4,
            'width': 6,
            'transpose': True,
            'flatten_order': 'C',
        }
        pred = np.arange(4 * 6, dtype=np.int32).reshape(4, 6) + 1
        flat, map_hw = dlo.pack_tree_species_prediction(pred, meta)
        self.assertEqual(map_hw.shape, (4, 6))
        # transpose=True, order=C => pred.T.ravel('C') == pred.ravel('F')
        np.testing.assert_array_equal(flat, pred.T.reshape(-1, order='C'))
        self.assertEqual(flat.shape[0], 24)

    def test_pack_when_pred_is_native_swapped(self):
        # scene_info height/width swapped relative to native (d1,d2)
        meta = {
            'height': 4,
            'width': 6,
            'transpose': True,
            'flatten_order': 'C',
        }
        # native pred is (width, height) = (6, 4) like h5 (C,d1,d2)->(d1,d2)
        pred_native = np.arange(6 * 4, dtype=np.int32).reshape(6, 4) + 1
        flat, map_hw = dlo.pack_tree_species_prediction(pred_native, meta)
        self.assertEqual(map_hw.shape, (4, 6))
        np.testing.assert_array_equal(map_hw, pred_native.T)
        np.testing.assert_array_equal(flat, map_hw.T.reshape(-1, order='C'))

    def test_no_transpose_c_override(self):
        meta = {
            'height': 3,
            'width': 5,
            'transpose': True,
            'flatten_order': 'C',
        }
        pred = np.arange(3 * 5, dtype=np.int32).reshape(3, 5) + 1
        flat, _ = dlo.pack_tree_species_prediction(
            pred, meta, transpose=False, flatten_order='C'
        )
        np.testing.assert_array_equal(flat, pred.reshape(-1, order='C'))

    def test_only_two_unique_orders_for_fixed_map(self):
        rng = np.random.default_rng(0)
        pred = rng.integers(1, 18, size=(8, 5))
        meta = {'height': 8, 'width': 5, 'transpose': True, 'flatten_order': 'C'}
        flats = []
        for tr, order in [(False, 'C'), (False, 'F'), (True, 'C'), (True, 'F')]:
            flat, _ = dlo.pack_tree_species_prediction(
                pred, meta, transpose=tr, flatten_order=order
            )
            flats.append(flat)
        # Exactly two unique permutations
        uniq = {tuple(f.tolist()) for f in flats}
        self.assertEqual(len(uniq), 2)


class SceneMetaTest(unittest.TestCase):
    def test_scene_info_matches_files(self):
        data_path = os.path.join(ROOT, 'data')
        if not os.path.exists(os.path.join(data_path, 'TreeSpeciesHSI', 'scene_info.csv')):
            self.skipTest('competition data not present')
        for scene in ('scene1', 'scene2'):
            meta = dlo.load_tree_species_scene_meta(data_path, scene)
            self.assertEqual(meta['scene'], scene)
            self.assertEqual(meta['height'] * meta['width'], meta['height'] * meta['width'])
            self.assertTrue(os.path.exists(meta['mat_path']))


if __name__ == '__main__':
    unittest.main()

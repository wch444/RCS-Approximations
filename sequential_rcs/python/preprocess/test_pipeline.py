"""Contract tests for the canonical sequential-RCS preprocessing entry point."""

from __future__ import annotations

import unittest

from . import implementation, pipeline


class SequentialRcsPreprocessEntrypointTests(unittest.TestCase):
    def test_entrypoint_exports_the_single_versioned_implementation(self) -> None:
        self.assertEqual(pipeline.PROTOCOL, "expedia_no_click_filter_no_position_v1")
        self.assertIs(pipeline.build_from_h5, implementation.build_from_h5)
        self.assertEqual(pipeline.MATRIX_COLUMN_COUNT, 20)
        self.assertEqual(pipeline.BASE_COLUMNS, implementation.BASE_COLUMNS)
        self.assertEqual(pipeline.MODEL_FEATURE_COLUMNS, pipeline.BASE_COLUMNS[7:])

    def test_project_implementation_has_no_click_or_position_filter(self) -> None:
        source = implementation.prepare_site
        self.assertEqual(implementation.MATRIX_COLUMN_COUNT, 20)
        self.assertNotIn("MAX_QUERY_CLICKS", implementation.__dict__)
        self.assertNotIn("POSITION_BIN_WIDTH", implementation.__dict__)
        self.assertIs(pipeline.prepare_site, source)


if __name__ == "__main__":
    unittest.main()

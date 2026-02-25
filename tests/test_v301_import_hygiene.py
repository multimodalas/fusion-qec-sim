"""
Import hygiene tests for v3.0.1.

Ensures that core decoding imports do not transitively pull in
bench, qudit, nonbinary, or analysis modules unless explicitly used.
"""

import sys


class TestCoreDecoderIsolation:
    """Importing core decoder must not import new v3.0.1 packages."""

    def _clear_modules(self, prefixes):
        for prefix in prefixes:
            mods = [k for k in sys.modules if k.startswith(prefix)]
            for m in mods:
                del sys.modules[m]

    def test_decoder_does_not_import_bench(self):
        self._clear_modules(["src.bench"])
        import src.decoder  # noqa: F401
        bench_mods = [k for k in sys.modules if k.startswith("src.bench")]
        assert bench_mods == []

    def test_decoder_does_not_import_qudit(self):
        self._clear_modules(["src.qudit"])
        import src.decoder  # noqa: F401
        qudit_mods = [k for k in sys.modules if k.startswith("src.qudit")]
        assert qudit_mods == []

    def test_decoder_does_not_import_analysis(self):
        self._clear_modules(["src.analysis"])
        import src.decoder  # noqa: F401
        analysis_mods = [k for k in sys.modules if k.startswith("src.analysis")]
        assert analysis_mods == []

    def test_decoder_does_not_import_nonbinary(self):
        self._clear_modules(["src.nonbinary"])
        import src.decoder  # noqa: F401
        nb_mods = [k for k in sys.modules if k.startswith("src.nonbinary")]
        assert nb_mods == []


class TestBenchIsolation:
    """Bench import must not import qudit or nonbinary."""

    def _clear_modules(self, prefixes):
        for prefix in prefixes:
            mods = [k for k in sys.modules if k.startswith(prefix)]
            for m in mods:
                del sys.modules[m]

    def test_bench_does_not_import_nonbinary(self):
        self._clear_modules(["src.nonbinary"])
        import src.bench  # noqa: F401
        nb_mods = [k for k in sys.modules if k.startswith("src.nonbinary")]
        assert nb_mods == []

    def test_bench_does_not_import_analysis(self):
        self._clear_modules(["src.analysis"])
        import src.bench  # noqa: F401
        analysis_mods = [k for k in sys.modules if k.startswith("src.analysis")]
        assert analysis_mods == []

"""
webapp/webapp_config.py

Feature flags for the demo web app.

Set a flag to False and the corresponding feature disappears completely: the
backend stops exposing its endpoint and the UI never renders its controls.
The app falls back to the plain upload -> verdict -> explanation flow.
"""

from __future__ import annotations

# --- Live behaviour-preserving attack demo ---------------------------------
# Adds an "Attack this sample" panel. The analysed graph is re-scored after a
# behaviour-preserving transformation, showing whether the verdict survives and
# how the adaptive gate redistributes its weights.
ENABLE_ATTACK_DEMO = False

# --- Side-by-side classical baseline ---------------------------------------
# Scores the same trace with the TF-IDF + tree-ensemble baseline of the
# thesis's classical paradigm, clean and under attack. Requires the baseline
# artefact (see webapp/README.md); the feature stays dark if it is absent.
ENABLE_BASELINE_RACE = False

# Number of analysed graphs kept in memory so they can be re-scored under
# attack without re-running enrichment. Enriched graphs are large; keep small.
GRAPH_CACHE_SIZE = 4

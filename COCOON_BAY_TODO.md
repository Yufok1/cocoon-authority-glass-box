# Cocoon Bay — status + TODO (checkpoint 2026-06-12)

The cocoon stack now has a complete, **tested** vertical slice:
**train → compile → illuminate → see-live.** Core is stable; items below are enhancements
/ not-yet-stabilized work to finish later.

## DONE + tested locally (stable)
- **Trainer load fix** (`mira_kite_infinite_trainer.py`) — `load_cocoon` now registers the
  module in `sys.modules` before `exec_module`, so the cocoon's `@dataclass`es resolve. The
  trainer loads, trains (real losses), grows vocab/knowledge, and exports a retained cocoon.
- **`compile_bay.py`** — merge N cocoons → ONE ensemble cocoon superfile. Verified: 2 cocoons
  → 4-organism ensemble that loads + retains; artifact PARITY with a standalone cocoon is exact
  (same 15 embedded blobs incl. README/drone/TMRL). A cocoon-of-cocoons is still a cocoon.
- **`cocoon_illumination.py`** — tiered causal self-knowledge, ported from the Butterfly
  `alliance_warfare` illumination engine. Tier EMERGES from compilation (more source cocoons =
  higher civilization; a bay merge = hegemony = omniscience). Reads the cocoon's OWN cascade
  record (knowledge-web relations + training logs). Access control faithful (single cocoon
  denied `all`, granted `self`).
- **`cocoon_graph.py`** — lean LIVE neural-network feed (D3 v7, ~5KB page). Polls
  `cocoon_graph.json`, smooth joins, 3 palettes (keys 1/2/3), cheap refraction (new nodes flash),
  minimal cosmetics for low latency, drillable (click → neighbors → traverse). The trainer emits
  the feed each cycle, so a running cocoon grows the graph live.

## TODO / not yet stabilized (finish later)
- [ ] **Mixed-dim merge**: `compile_bay` v1 merges SAME-dim cocoons. 30-dim legacy + 34-dim
      commerce hyper-learners need `max_input_dim` observation padding at the ensemble forward.
- [ ] **`get_events_by_component`**: the `self` illumination query returns `[]` on base cocoons
      (the component filter doesn't match the cocoon's training-log tagging) — align the filter.
- [ ] **Composite README regen**: `compile_bay` should regenerate the megafile's embedded
      `_README_B64` to document the novel N-organism / provenance / voting interactions.
- [ ] **Dispatch-to-jobs**: route specific organisms (by `@cN` provenance) to specific jobs — a
      routing layer over the `council` fan-out faculty.
- [ ] **Serve the feed from the Authority**: a `/api/cocoon_graph` route in
      `mira_kite_authority.py` instead of a file, for the HF Space + live-from-Authority streaming.
- [ ] **dreamer-oracle-gate governance**: gate the mega-cocoon's autonomous build/act intents
      (TALK/PLAN/BUILD) so "full autonomy" stays honest.
- [ ] **Richer GUI**: layer more `causation_explorer` features (filters, binder panels, event
      stream) onto the lean base.
- [ ] **MJPEG glass dashboard** (WIP in `mira_kite_authority.py` / `cocoon_glass_phone.py` /
      `Dockerfile`): headless `SDL_VIDEODRIVER=dummy` + MJPEG (HF bans VNC); test locally first.

## How to run the living mind
```
python mira_kite_infinite_trainer.py --cocoon <cocoon.py> --no-web --output-dir live_run
# open cocoon_mind.html (point DATA_URL at live_run/cocoon_graph.json)
```

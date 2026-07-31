# Integration handoff

_Completed in Phase 15. Maintained incrementally so the list is accurate rather
than reconstructed at the end._

This experimental branch (`zy_CNN`) is **not** merged into `main`. If the
experiment succeeds, selected finished files are transferred to a **new
integration branch created from the latest `main`**. That branch is not created
or switched to as part of this work.

## Transfer candidates (provisional, as of Phase 2)

| Path | Notes |
| --- | --- |
| `src/farm_pest_ai/scopes.py` | Self-contained; no dependencies beyond the standard library |
| `src/farm_pest_ai/config.py` | Requires `pyyaml` |
| `src/farm_pest_ai/logging_config.py` | Standard library only |
| `src/farm_pest_ai/reproducibility.py` | Optional imports; degrades without torch |
| `src/farm_pest_ai/cli.py` | Depends on the modules above |
| `configs/*.yaml` | Path defaults may need adjusting for the integration layout |
| `tests/*` | Must be rerun against `main`'s dependency versions |
| `pyproject.toml` | Reconcile with `main`'s existing build configuration |

## Experimental-only (do not transfer)

- `docs/STATUS.md` and `docs/PHASES.md` — specific to this phase-gated workflow
- Scratch analysis scripts kept outside the repository
- Any checkpoint or artifact produced for comparison rather than deployment

## To record before handoff

- Dependencies that must be reconciled with `main`
- Configuration changes required during integration
- Tests that must be rerun
- Model and artifact migration requirements
- Which `.gitignore` additions are still needed

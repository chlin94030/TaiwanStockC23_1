Alpha Radar V22.4.1 One-Click — Slim Full Package

Purpose
- Complete deployable package with fewer than 100 files.
- Keeps runtime code, one-click validation, data preparation, checkpoint/resume,
  report export, optional GitHub Actions background workflow, and current V22.4.1 docs.
- Removes historical release notes, old QA snapshots, pytest source files,
  development-only requirements, caches and lock files.

Deploy
1. Upload all files/folders in this ZIP to the repository root.
2. Do not rename app.py, requirements.txt, config.json, or the .github folder.
3. Existing data/ may be retained when upgrading; for a clean repository the empty
   data structure will be created automatically when the app runs.
4. Start Streamlit from app.py.

Validation
- Package file count is intentionally below 100.
- Python sources are compile-checked before release.
- This slim package includes the V22.4.1 derived-pool lock fix; strategy logic is otherwise unchanged.

# Repository TODOs

## Missing Features
- [ ] **CoreML Export**: Add class labels to CoreML export in `src/core/export/coreml_exporter.py`.
- [ ] **Dataset ID**: Pass `dataset_id` correctly in `src/core/export/exporter.py` instead of using "unknown".

## Refactoring Opportunities
- [ ] **Dataset Builder**: Refactor `scripts/build_otoscopic_dataset.py` to fully utilize `src/core/data/dataset_builder.py` to avoid code duplication.
- [ ] **Error Handling**: Improve error handling in `src/core/export/exporter.py` (e.g., `_get_git_commit` swallows exceptions).

## Code Cleanup
- [ ] **Placeholders**: Address `pass` statements in `src/core/export/coreml_exporter.py` (classification labels).

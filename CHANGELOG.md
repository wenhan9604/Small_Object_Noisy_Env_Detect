# Changelog

## [2025-04-09]

### To Do
- Implement data loader for dota-h.
- Implement trainer for `kjrd_net` in `trainers/trainer_kjrd_net.py`.
- Modify `icb` and `kjrd` models to include encoder and rcan.

### Added
- `/models/` folder containing implementations for `ffa_net`, `icb`, and `kjrd` models.
- `/datasets/dotah_ffa_net_dataset.py` for loading hazy and clear datasets used to train/test FFA-Net.
- `/temp_adhoc_scripts/` folder for miscellaneous scripts.
- `/trainers/` folder with training logic for `ffa_net` or `kjrd_net`
- `train.py`: script to run training for `ffa_net` or `kjrd_net`.

### Changed
- Moved `RCAN.py` into `/models/`.
- Relocated `dotah.py` into `/temp_adhoc_scripts/`.

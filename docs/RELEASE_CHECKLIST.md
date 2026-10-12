# Before public paper release

1. Copy the original `protocol_no2.json`, `protocol_o3.json`, `protocol_pm25.json` from the server's `f3_direct_ablation_final_strict_v1/D3/` into `configs/reference/D3/`. Confirm their actual location; do not fabricate metadata.
2. Recover the final preprocessing code and exact covariate schema. The legacy recovered `zh.py` is not included as a claimed final preprocessing entry.
3. Add the processed data or a documented retrieval route and the permitted example needed to run a preflight/trial. Record the exact original software versions and GPU environment.
4. Recover the final spatial-anomaly and station-pair contrast generating script. The captured latest summaries use a zero-anomaly/zero-contrast reference; the earlier `spatial_skill_analysis.py` uses a different SpatialMean diagnostic and cannot silently replace that script.
5. Add final station-time prediction CSVs and the matched HM/RM seed summaries, with mappings to manuscript tables. Include model weights or their download link when supporting inference without retraining. Keep preliminary or mismatched run outputs out of the release.
6. Add a clear license, English usage documentation and English source comments for C&G submission. Preserve an archival source hash record. The RM launcher currently checks original source hashes: translating comments or changing paths changes those hashes. Record the release changes and consistently update the release integrity checks; retain provenance validation rather than deleting checks.
7. Replace server-specific defaults with configurable portable paths or document explicit command-line/environment overrides for every experiment. Run a preflight and a small configuration trial from a clean checkout, then verify reproduction of selected paper tables from the original saved predictions.
8. Upload the extracted file tree to GitHub. A repository containing only a ZIP file does not satisfy the current C&G code-repository requirements. Add the public repository link to the manuscript when the release is complete.

Sources consulted: the actual recovered Python files and the official Computers & Geosciences description at https://shop.elsevier.com/journals/computers-and-geosciences/0098-3004 (checked 2026-10-12).

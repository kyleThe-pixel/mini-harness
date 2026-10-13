# Adapter packages for the 3050 benchmark harness.
# Every adapter module exposes:
#   run(model, run_dir, sampler, cfg) -> dict
# where the dict has primary_label / primary_value plus extras,
# and optionally output_file (filename inside run_dir).

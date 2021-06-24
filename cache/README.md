# Summary of runs:

1. `30kg_payload`: Baseline, with 30 kg payload and standard assumptions.
2. `10kg_payload`: Changed to use 10 kg payload and standard assumptions.
3. `10kg_payload_continuous_power`: 10 kg payload w/ continuous payload power (100 W const., instead of 500 W / 150 W during day / night.)
4. `10kg_payload_no_cycling`: 10 kg payload with no altitude cycling (i.e. `allow_trajectory_optimization=False`)
5. `10kg_payload_sunpower`: 10 kg payload w/ Sunpower cell assumptions (changed `solar_cell_efficiency`, `rho_solar_cells`, and set `solar_area_fraction < 0.60`.)
6. `10kg_payload_ascent`: 10 kg payload w/ Ascent cell assumptions (changed `solar_cell_efficiency`, `rho_solar_cells`, and set `solar_area_fraction < 0.80`.)
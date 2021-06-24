# Summary of runs:

1. `30kg_payload`: Baseline, with 30 kg payload and standard assumptions.
2. `10kg_payload`: Changed to use 10 kg payload and standard assumptions.
3. `10kg_payload_continuous_power`: 10 kg payload w/ continuous payload power (100 W const., instead of 500 W / 150 W during day / night.)
4. `10kg_payload_no_cycling`: 10 kg payload with no altitude cycling (i.e. `allow_trajectory_optimization=False`)
from design_opt import *
from aerosandbox.visualization.carpet_plot_utils import time_limit, patch_nans
import aerosandbox.numpy as np

cache_suffix = "_10kg_payload"


def run_sweep():
    latitudes = np.linspace(-80, 80, 15)
    day_of_years = np.linspace(0, 365, 30)
    spans = []
    days = []
    lats = []
    num = 0
    for lat_val in latitudes:
        for day_val in day_of_years:
            print("\n".join([
                "-" * 50,
                f"latitude: {lat_val}",
                f"day of year: {day_val}",
            ]))
            opti.set_value(latitude, lat_val)
            opti.set_value(day_of_year, day_val)

            try:
                with time_limit(60 if num < 5 else 20):
                    sol = opti.solve()
                opti.set_initial(opti.value_variables())
                opti.set_initial(opti.lam_g, sol.value(opti.lam_g))

                lats.append(lat_val)
                days.append(day_val)
                spans.append(sol.value(wing_span))

            except Exception as e:
                print(e)

            num += 1

    np.save("cache/lats" + cache_suffix, lats)
    np.save("cache/days" + cache_suffix, days)
    np.save("cache/spans" + cache_suffix, spans)

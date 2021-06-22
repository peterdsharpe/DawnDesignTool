from aerosandbox.modeling.interpolation import InterpolatedModel
import aerosandbox.numpy as np
import pathlib
import pandas as pd
import matplotlib.pyplot as plt



path = str(
    pathlib.Path(__file__).parent.absolute()
)

strat_pd = pd.read_csv(path + '/cache/strat-height.csv')
strat_data = strat_pd.values
latitude = strat_pd.Latitude
height = strat_pd.Altitude

strat_model = InterpolatedModel({'latitude': latitude},
                                              height, 'bspline')
offset_value = 2500
lats = np.linspace(-90, 90)
min_cruise_alt = strat_model({'latitude': lats}) * 1000 + offset_value

plt.plot(lats, min_cruise_alt)



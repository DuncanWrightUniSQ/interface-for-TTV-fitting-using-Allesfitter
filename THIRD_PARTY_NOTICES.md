# Third-party notices

## Allesfitter

This interface builds on and interoperates with workflows from
`allesfitter`, which is distributed under the MIT License.

Allesfitter creators:

- Maximilian Guenther <maximilian.guenther@esa.int>
- Tansu Daylan <tansu@wustl.edu>

When using this project for scientific work, please also cite the appropriate
Allesfitter papers and documentation for the underlying modelling/fitting
methods used in your analysis.

## Scientific Python dependencies

This project is released under the MIT License, but its third-party
dependencies remain subject to their own licences. Important direct scientific
and visualisation dependencies include:

| Software | Use in this project | Upstream licence |
| --- | --- | --- |
| [BATMAN](https://github.com/lkreidberg/batman) | Analytic transit light-curve models | [GNU GPLv3](https://github.com/lkreidberg/batman/blob/master/LICENSE.txt) |
| [emcee](https://github.com/dfm/emcee) | Ensemble MCMC sampling | [MIT](https://github.com/dfm/emcee/blob/main/LICENSE) |
| [Wōtan](https://github.com/hippke/wotan) | Time-series detrending and flattening | [MIT](https://github.com/hippke/wotan/blob/master/LICENSE) |
| [Plotly.py](https://github.com/plotly/plotly.py) | Interactive plots | [MIT](https://github.com/plotly/plotly.py/blob/main/LICENSE.txt) |
| [Lightkurve](https://github.com/lightkurve/lightkurve) | TESS/Kepler light-curve support | [MIT](https://github.com/lightkurve/lightkurve/blob/main/LICENSE) |
| [REBOUND](https://github.com/hannorein/rebound) | N-body orbital and physical TTV modelling | [GNU GPLv3](https://github.com/hannorein/rebound/blob/main/LICENSE) |

The dependency list in `pyproject.toml` and `requirements.txt` includes other
open-source packages under their respective licences. Installing this project
does not change those upstream licence terms. When publishing scientific
results, cite the relevant software papers requested by each upstream project.

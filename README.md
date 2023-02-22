# The effect of weight loss following 18 months of lifestyle intervention on brain age assessed with resting-state functional connectivity

Python implementation of the analysis from Levakov et al. 2023 paper, "The effect of weight loss following 18 months 
of lifestyle intervention on brain age assessed with resting-state functional connectivity".

## Dependencies
- Python 3.7+
- Python packages: numpy, scipy, pandas, matplotlib, seaborn, scikit-learn, statsmodels


## Scripts description:
The following scripts are used for the analysis, in the order they should be run
* `model_training_and_validation.py` - training and validation of the brain age prediction model
* `brain_age_measures.py` - calculating brain age measures (brain age attenuation, and delta brain age)
* `brain_age_associations.py` - testing the association between brain age and various clinical measures

* Additional utility scripts:
    * `plot_utils.py` - utility functions for plotting
    * `brain_age_utils.py` - plotting functions for calculating brain age measures

## Citing

If you use this code, please cite:

    Levakov, G., Kaplan, A., Yaskolka Meir, A., Rinott, E., Tsaban, G., Zelicha, H., Bluher, M., Ceglarek, 
    U., Stumvoll, M., Shelef, I. and Avidan, G., (2023), “The effect of weight loss following 18 months of lifestyle 
    intervention on brain age assessed with resting-state functional connectivity”, eLife, (under review).

### Issues / questions
Please [raise an issue](https://github.com/GidLev/functional_brain_aging/issues) if you have any questions/ comments regarding the code.

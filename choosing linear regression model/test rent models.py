import pandas as pd
import numpy as np
import statsmodels.api as sm

# Import data
rentals = pd.read_csv('rentals.csv')

# Explore and test out models below:
model1 = sm.OLS.from_formula('rent ~ bedrooms + size_sqft + has_washer_dryer', data= rentals).fit()

model2 = sm.OLS.from_formula('rent ~ bedrooms + size_sqft + has_washer_dryer + borough', data= rentals).fit()


# compare models based on adjusted R-squared
print(model1.rsquared_adj)
print(model2.rsquared_adj)
print('\n')
# choose model2 - higher adjusted R-squared


# compare the models using F-test
from statsmodels.stats.anova import anova_lm
anova_results = anova_lm(model1, model2)
print(anova_results)
print('\n')
# assuming we are using a significance threshold of 0.05, Pr(>F) is way smaller. So we reject the null hypothesis and choose model2 over model1 and conclude that the coefficient of borough is not zero and it improves the model.


# compare the models using AIC/BIC
print(model1.aic)
print(model2.aic)
print('\n')
print(model1.bic)
print(model2.bic)
# for AIC, choose model2. for BIC, choose model2 - lower AIC/BIC

# all of the above tests agree in terms of what is considered 'best' model which is model2.

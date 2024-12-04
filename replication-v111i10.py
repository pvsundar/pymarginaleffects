# Warning
# Some numerical results are slightly different in Python and R because the
# Python version does not use back-transformation to compute confidence
# intervals and p values in GLM models.  Other minor differences may occur due
# to do different row ordering.  These differences are inconsequential in terms
# of the substantive results.

import polars as pl
import numpy as np
from marginaleffects import *
import statsmodels.formula.api as smf

dat = pl.read_csv("https://marginaleffects.com/data/impartiality.csv") \
  .with_columns(pl.col("impartial").cast(pl.Int8))

# load locally from the supplemental information package
# dat = pl.read_csv("impartiality.csv") \
#   .with_columns(pl.col("impartial").cast(pl.Int8))

m = smf.logit(
  "impartial ~ equal * democracy + continent",
  data = dat.to_pandas()
).fit()

p = predictions(m)
p

p["estimate"][:4]
p.filter(pl.col("estimate") == pl.min("estimate"))

# type = "link" is not availble in Python
# predictions(m, type = "link", newdata = dat.head(2))

predictions(m, newdata = "mean")

datagrid(model = m, democracy = dat["democracy"].unique(), equal = [30, 90])

predictions(m,
  newdata = datagrid(democracy = dat["democracy"].unique(), equal = [30, 90])
).select("democracy", "equal", "estimate", "conf_low", "conf_high")

avg_predictions(m)
np.mean(m.predict())

predictions(m, by = "democracy")

plot_predictions(m, by = ["democracy", "continent"])

# vcov="HC3" and vcov=~continent are not yet available in Python
avg_predictions(m,
  by = "democracy",
  conf_level = .99,
  hypothesis = .4
)

# inferences() is not yet available in Python
hypotheses(m, hypothesis = "b4 = b3")

avg_predictions(m, by = "democracy")

avg_predictions(m,
  by = "democracy",
  hypothesis = "pairwise")

predictions(m,
  by = "democracy",
  hypothesis = "b1 = b0 * 2")

predictions(m,
  by = "democracy",
  hypothesis = "b1 = b0 * 2",
  equivalence = [-.2, .2]) \
  .select("term", "estimate", "std_error", "p_value_equiv")

comparisons(m, variables = "democracy")

avg_comparisons(m)

comparisons(m, variables = "democracy")["estimate"].mean()

dat_lo = dat.with_columns(pl.lit("Autocracy").alias("democracy"))
dat_hi = dat.with_columns(pl.lit("Democracy").alias("democracy"))
pred_lo = m.predict(dat_lo.to_pandas())
pred_hi = m.predict(dat_hi.to_pandas())
np.mean(pred_hi - pred_lo)

dat_lo = dat.with_columns(pl.lit("Autocracy").alias("democracy"))
dat_hi = dat.with_columns(pl.lit("Democracy").alias("democracy"))
pred_lo = m.predict(dat_lo.to_pandas())
pred_hi = m.predict(dat_hi.to_pandas())
np.mean(pred_hi) / np.mean(pred_lo)

avg_comparisons(m, variables = "democracy", comparison = "ratio")

avg_comparisons(m,
  comparison = "lnor",
  transform = np.exp)

## lambda functions are not supported by the `comparison` argument yet
# avg_comparisons(m,
#   variables = "equal",
#   comparison = lambda hi, lo: hi.mean() / lo.mean())

comparisons(m,
  by = "democracy",
  variables = {"equal": [30, 90]})


plot_comparisons(m,
  by = "democracy",
  variables = {"equal": [30, 90]}).show()


cmp = comparisons(m,
  by = "democracy",
  variables = {"equal": [30, 90]},
  hypothesis = "pairwise")
cmp

slopes(m, variables = "equal", newdata = datagrid(equal = [25, 50]))

avg_slopes(m, variables = "equal")

slopes(m, variables = "equal", newdata = "mean")

slopes(m, variables = "equal", newdata = "median", by = "democracy")

avg_slopes(m, variables = "equal", slope = "eyex")




# Python section of the paper
import pandas as pd
import statsmodels.formula.api as smf
from marginaleffects import avg_predictions, slopes
dat = pd.read_csv("https://marginaleffects.com/data/impartiality.csv")
mod = smf.logit("impartial ~ equal * democracy + continent", data=dat).fit()
p = avg_predictions(mod, by="continent")
p
s = slopes(mod, variables="equal", newdata="mean")
s

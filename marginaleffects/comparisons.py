# TODO: Sanitize data: pandas, polars, or numpy array
from .by import *
from .utils import *
from .sanity import *
from .hypothesis import *
from .uncertainty import *
from .sanitize_variables import *
import polars as pl
import pandas as pd
import numpy as np
import patsy
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm


estimands = {
    "difference": lambda hi, lo, eps, x, y: hi - lo,
    "differenceavg": lambda hi, lo, eps, x, y: np.array([np.mean(hi - lo)]),
    # "differenceavgwts": lambda hi, lo, w: (hi * w).sum() / w.sum() - (lo * w).sum() / w.sum(),

    "dydx": lambda hi, lo, eps, x, y: (hi - lo) / eps,
    "eyex": lambda hi, lo, eps, x, y: (hi - lo) / eps * (x / y),
    "eydx": lambda hi, lo, eps, x, y: ((hi - lo) / eps) / y,
    "dyex": lambda hi, lo, eps, x, y: ((hi - lo) / eps) * x,

    "dydxavg": lambda hi, lo, eps, x, y: np.array([np.mean((hi - lo) / eps)]),
    "eyexavg": lambda hi, lo, eps, x, y: np.array([np.mean((hi - lo) / eps * (x / y))]),
    "eydxavg": lambda hi, lo, eps, x, y: np.array([np.mean((hi - lo) / eps) / y]),
    "dyexavg": lambda hi, lo, eps, x, y: np.array([np.mean(((hi - lo) / eps) * x)]),
    # "dydxavgwts": lambda hi, lo, eps, w: (((hi - lo) / eps) * w).sum() / w.sum(),
    # "eyexavgwts": lambda hi, lo, eps, y, x, w: (((hi - lo) / eps) * (x / y) * w).sum() / w.sum(),
    # "eydxavgwts": lambda hi, lo, eps, y, x, w: ((((hi - lo) / eps) / y) * w).sum() / w.sum(),
    # "dyexavgwts": lambda hi, lo, eps, x, w: (((hi - lo) / eps) * x * w).sum() / w.sum(),

    "ratio": lambda hi, lo, eps, x, y: hi / lo,
    "ratioavg": lambda hi, lo, eps, x, y: np.array([np.mean(hi) / np.mean(lo)]),
    # "ratioavgwts": lambda hi, lo, w: (hi * w).sum() / w.sum() / (lo * w).sum() / w.sum(),

    "lnratio": lambda hi, lo, eps, x, y: np.log(hi / lo),
    "lnratioavg": lambda hi, lo, eps, x, y: np.array([np.log(np.mean(hi) / np.mean(lo))]),
    # "lnratioavgwts": lambda hi, lo, w: np.log((hi * w).sum() / w.sum() / (lo * w).sum() / w.sum()),

    "lnor": lambda hi, lo, eps, x, y: np.log((hi / (1 - hi)) / (lo / (1 - lo))),
    "lnoravg": lambda hi, lo, eps, x, y: np.log((np.mean(hi) / (1 - np.mean(hi))) / (np.mean(lo) / (1 - np.mean(lo)))),
    # "lnoravgwts": lambda hi, lo, w: np.log(((hi * w).sum() / w.sum() / (1 - (hi * w).sum() / w.sum())) / ((lo * w).sum() / w.sum() / (1 - (lo * w).sum() / w.sum()))),

    "lift": lambda hi, lo, eps, x, y: (hi - lo) / lo,
    "liftavg": lambda hi, lo, eps, x, y: np.array([(np.mean(hi) - np.mean(lo)) / np.mean(lo)]),

    "expdydx": lambda hi, lo, eps, x, y: ((np.exp(hi) - np.exp(lo)) / np.exp(eps)) / eps,
    "expdydxavg": lambda hi, lo, eps, x, y: (((np.exp(hi) - np.exp(lo)) / np.exp(eps)) / eps).mean(),
    # "expdydxavgwts": lambda hi, lo, eps, w: ((((np.exp(hi) - np.exp(lo)) / np.exp(eps)) / eps) * w).sum() / w.sum(),
}



    
def comparisons(
        model,
        variables = None,
        newdata = None,
        comparison = "differenceavg",
        vcov = True,
        conf_int = 0.95,
        by = None,
        hypothesis = None,
        eps = 1e-4):
    """
    Comparisons Between Predictions Made With Different Regressor Values

    Predict the outcome variable at different regressor values (e.g., college graduates vs. others), and compare those predictions by computing a difference, ratio, or some other function. `comparisons()` can return many quantities of interest, such as contrasts, differences, risk ratios, changes in log odds, lift, 

    Parameters
    ----------
    * model : `statsmodels.formula.api` modelted model
    * conf_int : float
    * vcov : bool or string which corresponds to one of the attributes in `model`. "HC3" looks for an attributed named `cov_HC3`.
    * newdata : None, DataFrame or `datagrid()` call.
    * hypothesis : Numpy array for linear combinations. 
    * comparison : "difference", "differenceavg", "ratio", "ratioavg", "lnratio", "lnratioavg", "lnor", "lnoravg", "lift", "liftavg", "expdydx", "expdydxavg", "expdydxavgwts"
    * by : None, string, or list of strings
    """


    # sanity
    V = sanitize_vcov(vcov, model)
    newdata = sanitize_newdata(model, newdata)

    # after sanitize_newdata() 
    variables = sanitize_variables(variables=variables, model=model, newdata=newdata, comparison=comparison, eps=eps)

    xvar = None
    yvar = None
    def fun(coefs, v):
        lo = newdata.clone().with_columns(pl.Series(v.lo).alias(v.variable))
        hi = newdata.clone().with_columns(pl.Series(v.hi).alias(v.variable))
        k, lo = patsy.dmatrices(model.model.formula, lo.to_pandas())
        k, hi = patsy.dmatrices(model.model.formula, hi.to_pandas())
        lo = model.model.predict(coefs, lo)
        hi = model.model.predict(coefs, hi)
        est = estimands[comparison](hi = hi, lo = lo, eps = eps, x = xvar, y = yvar)
        if len(est) == newdata.shape[0]:
            out = newdata.with_columns(
                pl.Series(lo).alias("predicted_lo"),
                pl.Series(hi).alias("predicted_hi"),
                pl.Series(est).alias("estimate"),
                pl.Series([v.variable]).alias("term"),
                pl.Series([v.lab]).alias("contrast"),
            )
        else:
            out = pl.DataFrame({
                "term": [v.variable],
                "contrast": [v.lab],
                "estimate": est,
            })
        return out

    res = []
    for v in variables:
        tmp = fun(model.params, v)
        res.append(tmp)
        if vcov is not None and vcov is not False:
            g = lambda x: fun(x, v)
            J = get_jacobian(g, model.params.to_numpy())
            se = get_se(J, vcov)
            out = out.with_columns(pl.Series(se).alias("std_error"))
    for i, r in enumerate(res):
        for col in res[0].columns:
            if r[col].dtype is pl.Categorical:
                res[i] = r.with_columns(pl.col(col).cast(pl.Utf8))
    out = pl.concat(res)

    # if variabletype != "numeric" and comparison in ["dydx", "eyex", "eydx", "dyex"]:
    #     fun = estimands["difference"]
    # elif variabletype != "numeric" and comparison in ["dydxavg", "eyexavg", "eydxavg", "dyexavg"]:
    #     fun = estimands["differenceavg"]
    # else:
    #     fun = estimands[comparison]

    # uncetainty

    # uncertainty
    out = get_z_p_ci(out, model, conf_int=conf_int)

    # output
    out = sort_columns(out, by = by)
    return out
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

#TODO: Call in data, process it, and actually produce tilts.  
# Then add (lots) more rules.. carefully
# Finally, refine MF's per input if necessary


def z_to_universe(z):
    """
    Clip rolling z data to -2.5 through 2.5 so fuzzy inputs aren't 
    distorted by outliers
    """
    return np.clip(z, -2.5, 2.5)

# TODO: Starting w/ basic MF's accross the board, make custom ones per input later (Eg., CPI isn't so simple)
# Input MF's, using a standard Low, Med, High
def three_mf(name, var):
    """
    Build standard Ruspini-like partition:
      Low, Mid, High on a symmetric universe.
    """
    low = fuzz.trimf(var.universe, [-2.5, -2.5, 0])
    mid = fuzz.trimf(var.universe, [-1, 0, 1])
    high = fuzz.trimf(var.universe, [0, 2.5, 2.5])

    return {
        f"{name}_low": low,
        f"{name}_mid": mid,
        f"{name}_high": high
    }

# This is the output, tilt, mfs
def tilt_mfs(var):
    return {
        "under": fuzz.trimf(var.universe, [0.0, 0.0, 0.4]),
        "neutral": fuzz.trimf(var.universe, [0.3, 0.5, 0.7]),
        "over": fuzz.trimf(var.universe, [0.6, 1.0, 1.0]),
    }

def build_vym_fis(): # dividend yield
    """
    Inputs correspond to:
      - 10Y Yield      (negative)
      - CPI            (inverse U-shape)
      - GDP Growth     (negative)
      - HY Spread      (positive)
    """

    y10y = ctrl.Antecedent(np.linspace(-2.5, 2.5, 101), "y10y")
    cpi = ctrl.Antecedent(np.linspace(-2.5, 2.5, 101), "cpi")
    gdpg = ctrl.Antecedent(np.linspace(-2.5, 2.5, 101), "gdpg")
    spd = ctrl.Antecedent(np.linspace(-2.5, 2.5, 101), "hyspd")
    tilt = ctrl.Consequent(np.linspace(0, 1, 101), "tilt")
    tilt.defuzzify_method = 'mom'


    # define memberships
    for var, nm in [(y10y,"y10y"),(cpi,"cpi"),(gdpg,"gdpg"),(spd,"spd")]:
        m = three_mf(nm, var)
        for k,v in m.items(): var[k] = v
    
    for k,v in tilt_mfs(tilt).items():
        tilt[k] = v
    
    rules = []
    rules.append(ctrl.Rule(y10y["y10y_high"],tilt["under"]))
    rules.append(ctrl.Rule(y10y["y10y_low"],tilt["over"]))
    rules.append(ctrl.Rule(cpi["cpi_mid"],tilt["over"]))
    rules.append(ctrl.Rule(cpi["cpi_high"] | cpi["cpi_low"],tilt["under"]))
    rules.append(ctrl.Rule(gdpg["gdpg_low"],tilt["over"]))
    rules.append(ctrl.Rule(gdpg["gdpg_high"],tilt["under"]))
    rules.append(ctrl.Rule(spd["spd_high"],tilt["over"]))
    rules.append(ctrl.Rule(spd["spd_low"],tilt["under"]))
    rules.append(ctrl.Rule(gdpg["gdpg_high"] & spd["spd_low"],tilt["under"]))
    rules.append(ctrl.Rule(gdpg["gdpg_high"] & cpi["cpi_high"],tilt["under"]))
    # TODO: add more rules lol

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


def build_ivw_fis(): # Growth ETF
    """
    Inputs correspond to:
      - 10Y-2Y Treasury Spread    (positive)
      - Real Federal Funds Rate   (negative)
      - Corporate Profits Proxy (EPS YoY) (positive)
      - Capex    (positive)
    """

    spd   = ctrl.Antecedent(np.linspace(-2.5, 2.5, 101), "spread")
    rff   = ctrl.Antecedent(np.linspace(-2.5, 2.5, 101), "realff")
    eps   = ctrl.Antecedent(np.linspace(-2.5, 2.5, 101), "eps")
    capex = ctrl.Antecedent(np.linspace(-2.5, 2.5, 101), "capex")
    tilt  = ctrl.Consequent(np.linspace(0, 1, 101), "tilt")
    tilt.defuzzify_method = 'mom'


    # define memberships
    for var, nm in [(spd,"spd"),(rff,"rff"),(eps,"eps"),(capex,"capex")]:
        m = three_mf(nm, var) # define membership function degrees
        for k,v in m.items(): var[k] = v

    for k,v in tilt_mfs(tilt).items():
        tilt[k] = v

    # create list of rules using defined antecedents and consequent
    rules = []
    rules.append(ctrl.Rule(spd["spd_high"], tilt["over"])) # If spd is large then invest more heavily in growth
    rules.append(ctrl.Rule(spd["spd_low"], tilt["under"]))
    rules.append(ctrl.Rule(rff["rff_high"], tilt["under"]))  # If real fed funds rate is high, then underinvest in growth
    rules.append(ctrl.Rule(rff["rff_low"], tilt["over"]))
    rules.append(ctrl.Rule(eps["eps_high"], tilt["over"])) # If EPS(YOY) is high then invest more heavily in growth
    rules.append(ctrl.Rule(eps["eps_low"], tilt["under"]))
    rules.append(ctrl.Rule(capex["capex_high"], tilt["over"])) # If capex is high then invest more heavily in growth
    rules.append(ctrl.Rule(capex["capex_low"], tilt["under"]))
    rules.append(ctrl.Rule(eps["eps_low"] & spd["spd_low"], tilt["under"])) # If EPS is low AND spd is low then invest less heavily in growth
    # TODO: add more rules

    system = ctrl.ControlSystem(rules) 
    return ctrl.ControlSystemSimulation(system)



def build_pdp_fis(): # Momentum etf
    """
    Inputs correspond to:
      - Market Volatility (VIX)        (negative)
      - Liquidity Growth (M2 YoY)      (positive)
      - High-Yield Credit Spread       (negative)
      - Industrial Production YoY      (positive)
      - Consumer Sentiment             (positive)
    """
    vix = ctrl.Antecedent(np.linspace(-2.5,2.5,101),"vix")
    m2 = ctrl.Antecedent(np.linspace(-2.5,2.5,101), "m2")
    spd = ctrl.Antecedent(np.linspace(-2.5,2.5,101), "spd")
    inp = ctrl.Antecedent(np.linspace(-2.5,2.5,101), "inp")
    cnsm = ctrl.Antecedent(np.linspace(-2.5,2.5,101),"cnsm")
    tilt  = ctrl.Consequent(np.linspace(0, 1, 101), "tilt")
    tilt.defuzzify_method = 'mom'


    # define memberships
    for var, nm in [(vix,"vix"),(m2,"m2"),(spd,"spd"),(inp,"inp"),(cnsm,"cnsm")]:
        m = three_mf(nm, var) # define membership function degrees
        for k,v in m.items(): var[k] = v

    for k,v in tilt_mfs(tilt).items():
        tilt[k] = v

    # create list of rules using defined antecedents and consequent
    rules = []
    rules.append(ctrl.Rule(vix["vix_high"], tilt["under"]))
    rules.append(ctrl.Rule(vix["vix_low"], tilt["over"]))
    rules.append(ctrl.Rule(m2["m2_high"],tilt["over"]))
    rules.append(ctrl.Rule(m2["m2_low"],tilt["under"]))
    rules.append(ctrl.Rule(spd["spd_high"],tilt["under"]))
    rules.append(ctrl.Rule(spd["spd_low"],tilt["over"]))
    rules.append(ctrl.Rule(inp["inp_high"],tilt["over"]))
    rules.append(ctrl.Rule(inp["inp_low"],tilt["under"]))
    rules.append(ctrl.Rule(cnsm["cnsm_high"],tilt["over"]))
    rules.append(ctrl.Rule(cnsm["cnsm_low"],tilt["under"]))
    rules.append(ctrl.Rule(vix["vix_low"] & cnsm["cnsm_high"] & spd["spd_low"], tilt["over"]))
    rules.append(ctrl.Rule(vix["vix_high"] & spd["spd_high"],tilt["under"]))
    # TODO: add more rules

    system = ctrl.ControlSystem(rules) 
    return ctrl.ControlSystemSimulation(system)

def build_spx_fis(): # Beta ETF
    """
    Inputs correspond to:
      - Economic Policy Unscertanty     (negative)
        VIX                             (negative)
        Consumer Sentiment              (positive)
        M2 Growth                       (positive)
        NFP changes                     (positive)
    """

    esi = ctrl.Antecedent(np.linspace(-2.5,2.5,101),"esi")
    vix = ctrl.Antecedent(np.linspace(-2.5,2.5,101),"vix")
    cnsm = ctrl.Antecedent(np.linspace(-2.5,2.5,101),"cnsm")
    m2 = ctrl.Antecedent(np.linspace(-2.5,2.5,101),"m2")
    nfp = ctrl.Antecedent(np.linspace(-2.5,2.5,101),"nfp")
    tilt  = ctrl.Consequent(np.linspace(0, 1, 101), "tilt")
    tilt.defuzzify_method = 'mom'


    for var, nm in [(esi,"esi"),(vix,"vix"),(cnsm,"cnsm"),(m2,"m2"),(nfp,"nfp")]:
        m = three_mf(nm, var) # define membership function degrees
        for k,v in m.items(): var[k] = v

    for k,v in tilt_mfs(tilt).items():
        tilt[k] = v
    
    rules = []
    rules.append(ctrl.Rule(esi["esi_high"],tilt["under"]))
    rules.append(ctrl.Rule(esi["esi_low"],tilt["over"]))
    rules.append(ctrl.Rule(vix["vix_high"],tilt["under"]))
    rules.append(ctrl.Rule(vix["vix_low"],tilt["over"]))
    rules.append(ctrl.Rule(cnsm["cnsm_high"],tilt["over"]))
    rules.append(ctrl.Rule(cnsm["cnsm_low"],tilt["under"]))
    rules.append(ctrl.Rule(m2["m2_high"],tilt["over"]))
    rules.append(ctrl.Rule(m2["m2_low"],tilt["under"]))
    rules.append(ctrl.Rule(nfp["nfp_high"],tilt["over"]))
    rules.append(ctrl.Rule(nfp["nfp_low"],tilt["under"]))
    rules.append(ctrl.Rule(esi["esi_high"] & vix["vix_high"],tilt["under"]))
    rules.append(ctrl.Rule(m2["m2_low"] & cnsm["cnsm_high"],tilt["over"]))
    # TODO: add more rules

    system = ctrl.ControlSystem(rules) 
    return ctrl.ControlSystemSimulation(system)


def build_vflq_fis(): # Low Volitility Index
    """
    Inputs correspond to:
      - Effective Federal Funds Rate           (negative)
      - Corporate Bond Spreads                 (negative)
      - Retail Trading Volume                  (positive)
      - Money Supply (M2 Level or Growth)      (positive)
      - Non-Farm Payrolls (NFP)                (positive)
    """

    effr = ctrl.Antecedent(np.linspace(-2.5,2.5,101),"effr")
    cbspd = ctrl.Antecedent(np.linspace(-2.5,2.5,101),"cbspd")
    rtv = ctrl.Antecedent(np.linspace(-2.5,2.5,101),"rtv")
    m2 = ctrl.Antecedent(np.linspace(-2.5,2.5,101),"m2")
    nfp = ctrl.Antecedent(np.linspace(-2.5,2.5,101),"nfp")
    tilt  = ctrl.Consequent(np.linspace(0, 1, 101), "tilt")
    tilt.defuzzify_method = 'mom'


    for var, nm in [(effr,"effr"),(cbspd,"cbspd"),(rtv,"rtv"),(m2,"m2"),(nfp,"nfp")]:
        m = three_mf(nm, var) # define membership function degrees
        for k,v in m.items(): var[k] = v

    for k,v in tilt_mfs(tilt).items():
        tilt[k] = v

    rules = []
    rules.append(ctrl.Rule(effr["effr_high"], tilt["under"]))
    rules.append(ctrl.Rule(effr["effr_low"], tilt["over"]))
    rules.append(ctrl.Rule(cbspd["cbspd_high"],tilt["under"]))
    rules.append(ctrl.Rule(cbspd["cbspd_low"],tilt["over"]))
    rules.append(ctrl.Rule(rtv["rtv_high"],tilt["over"]))
    rules.append(ctrl.Rule(rtv["rtv_low"],tilt["under"]))
    rules.append(ctrl.Rule(m2["m2_high"],tilt["over"]))
    rules.append(ctrl.Rule(m2["m2_low"],tilt["under"]))
    rules.append(ctrl.Rule(nfp["nfp_high"],tilt["over"]))
    rules.append(ctrl.Rule(nfp["nfp_low"],tilt["under"]))
    # TODO: add more rules

    system = ctrl.ControlSystem(rules) 
    return ctrl.ControlSystemSimulation(system)


def build_sp5mv_fis(): # Volitility Index
    """
    Inputs correspond to:
      - Market Volatility (VIX)                 (positive)
      - Real GDP Growth                         (negative)
      - Credit Spreads                          (positive)
      - Consumer Sentiment                      (negative)
      - Unemployment Rate                       (positive)
    """

    vix = ctrl.Antecedent(np.linspace(-2.5,2.5,101),"vix")
    rgdp = ctrl.Antecedent(np.linspace(-2.5,2.5,101),"rgdp")
    cspd = ctrl.Antecedent(np.linspace(-2.5,2.5,101),"cspd")
    cnsm = ctrl.Antecedent(np.linspace(-2.5,2.5,101),"cnsm")
    unmpr = ctrl.Antecedent(np.linspace(-2.5,2.5,101),"unmpr")
    tilt  = ctrl.Consequent(np.linspace(0, 1, 101), "tilt")
    tilt.defuzzify_method = 'mom'


    for var, nm in [(vix,"vix"),(rgdp,"rgdp"),(cspd,"cspd"),(cnsm,"cnsm"),(unmpr,"unmpr")]:
        m = three_mf(nm, var) # define membership function degrees
        for k,v in m.items(): var[k] = v

    for k,v in tilt_mfs(tilt).items():
        tilt[k] = v

    rules = []
    rules.append(ctrl.Rule(vix["vix_high"],tilt["over"]))
    rules.append(ctrl.Rule(vix["vix_low"],tilt["under"]))
    rules.append(ctrl.Rule(rgdp["rgdp_high"], tilt["under"]))
    rules.append(ctrl.Rule(rgdp["rgdp_low"], tilt["over"]))
    rules.append(ctrl.Rule(cspd["cspd_high"],tilt["over"]))
    rules.append(ctrl.Rule(cspd["cspd_low"],tilt["under"]))
    rules.append(ctrl.Rule(cnsm["cnsm_high"],tilt["under"]))
    rules.append(ctrl.Rule(cnsm["cnsm_low"],tilt["over"]))
    rules.append(ctrl.Rule(unmpr["unmpr_high"],tilt["over"]))
    rules.append(ctrl.Rule(unmpr["unmpr_low"],tilt["under"]))
    rules.append(ctrl.Rule(cspd["cspd_high"] & unmpr["unmpr_low"],tilt["over"]))
    rules.append(ctrl.Rule(rgdp["rgdp_high"] & vix["vix_low"], tilt["under"]))
    rules.append(ctrl.Rule(cnsm["cnsm_low"] & rgdp["rgdp_low"],tilt["over"]))
    # TODO: add more rules

    system = ctrl.ControlSystem(rules) 
    return ctrl.ControlSystemSimulation(system)
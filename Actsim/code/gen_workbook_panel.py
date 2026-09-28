import numpy as np, pandas as pd, warnings, contextlib, io, logging
from dataclasses import replace
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import prism_sim as ps, prism_noext as px, prism_robust as rb
from actsim import ClaimSimulator

cfg0=replace(ps.CFG, channel="severity"); K=cfg0.K
BOOK=px.Book(claims_q0=2000.0)   # ~80k claims total, manageable + good detection

def gen_claims(seed, scenario, inflect, cfg, book=BOOK):
    native=ps.generate_panel(seed, scenario, inflect, cfg)
    exposure=native["exposure"]*(book.claims_q0/(cfg.freq0*1000.0))
    T,Kk=cfg.T,cfg.K
    g=ps._trend_path(cfg,scenario,cfg.g0,cfg.g1) if inflect else np.full(T,cfg.g0)
    drift=g.copy(); drift[0]=0.0
    lms=np.log(cfg.sev0)+np.cumsum(drift)
    pol=px._policies(lms,exposure,book,cfg)
    logging.getLogger("actsim").setLevel(logging.ERROR)
    with contextlib.redirect_stdout(io.StringIO()):
        sim=ClaimSimulator(pol, random_seed=px.ACTSIM_OFFSET+seed); sim.simulate_claims()
    cl=sim.claim_data.copy()
    pid=cl["policy_id"].to_numpy().astype(int)
    cl["quarter"]=pid//1000; cl["sev_class"]=(pid%1000)//100
    counts=np.bincount(cl["quarter"].to_numpy(),minlength=T)
    total=np.bincount(cl["quarter"].to_numpy(),weights=cl["amount"].to_numpy(float),minlength=T)
    sev=np.full(T,np.nan); nz=counts>0; sev[nz]=total[nz]/counts[nz]
    if not nz[0]: sev[0]=np.exp(lms[0])
    for t in range(1,T):
        if not nz[t]: sev[t]=sev[t-1]
    return cl, exposure, counts, total, sev, lms

def _refgen(i): return px.panel_actsim(960000+i,"none",False,cfg0,BOOK)["logsev"]

if __name__=="__main__":
    # reference for internal signal (filtered)
    with Pool(2) as p: refraw=p.map(_refgen, range(80))
    ref=rb.build_ref([rb.hampel(x) for x in refraw], cfg0, use_filter=True)
    cfg=replace(cfg0, sig_d=ref["ret_sd"])
    sd_raw=float(np.std(np.concatenate([np.diff(x) for x in refraw]))); cfg_raw=replace(cfg0, sig_d=sd_raw)
    # find a showcase gradual panel where PRISM fires around K+3..K+9
    chosen=None
    for seed in range(861000,861040):
        cl,exp,counts,total,sev,lms=gen_claims(seed,"gradual",True,cfg0)
        logsev=np.log(sev); flt=rb.hampel(logsev)
        s=rb.signal(flt,cfg,ref); haz=ps.sigmoid(ps.logit(cfg.h0)+cfg.beta_int*s)
        Ap=ps.bocpd(np.concatenate([[0.0],np.diff(flt)]),haz,cfg)
        Ab=ps.bocpd(np.concatenate([[0.0],np.diff(logsev)]),cfg_raw.h0,cfg_raw)
        fp=rb.first_A(Ap,cfg); fb=rb.first_A(Ab,cfg_raw)
        if fp is not None and K<=fp<=K+9 and Ab[4:].max()<0.45:
            chosen=(seed,cl,exp,counts,total,sev,logsev,Ap,Ab,fp,fb,s); break
    if chosen is None:  # fallback
        seed=861000; cl,exp,counts,total,sev,lms=gen_claims(seed,"gradual",True,cfg0)
        logsev=np.log(sev); flt=rb.hampel(logsev); s=rb.signal(flt,cfg,ref)
        haz=ps.sigmoid(ps.logit(cfg.h0)+cfg.beta_int*s)
        Ap=ps.bocpd(np.concatenate([[0.0],np.diff(flt)]),haz,cfg); Ab=ps.bocpd(np.concatenate([[0.0],np.diff(logsev)]),cfg_raw.h0,cfg_raw)
        fp=rb.first_A(Ap,cfg); fb=rb.first_A(Ab,cfg_raw)
        chosen=(seed,cl,exp,counts,total,sev,logsev,Ap,Ab,fp,fb,s)
    seed,cl,exp,counts,total,sev,logsev,Ap,Ab,fp,fb,s=chosen
    T=cfg.T
    starts=[(pd.Timestamp("2015-01-01")+pd.DateOffset(months=3*t)).date() for t in range(T)]
    # claims sheet
    cl=cl.reset_index(drop=True)
    claims=pd.DataFrame(dict(claim_id=np.arange(1,len(cl)+1),
        quarter=cl["quarter"].to_numpy(),
        accident_quarter_start=[starts[q] for q in cl["quarter"].to_numpy()],
        severity_class=pd.Series(cl["sev_class"].to_numpy()).map({0:"light",1:"medium",2:"heavy"}).to_numpy(),
        claim_amount=np.round(cl["amount"].to_numpy(float),2)))
    # quarterly sheet
    freq=counts/exp; logret=np.concatenate([[0.0],np.diff(np.log(sev))])
    quarterly=pd.DataFrame(dict(quarter=np.arange(T), quarter_start=starts,
        exposure=np.round(exp,1), claim_count=counts, total_loss=np.round(total,2),
        mean_severity=np.round(sev,2), frequency=np.round(freq,5),
        sev_logret=np.round(logret,5), is_inflection_onset=[q==K for q in range(T)]))
    # trend exhibit: fitted pre-inflection baseline (first 16 q), extrapolated
    t=np.arange(T); B=cfg.B
    b1,b0=np.polyfit(t[:B], np.log(sev)[:B], 1)
    fitted=np.exp(b0+b1*t)
    dev=sev/fitted-1.0
    trend=pd.DataFrame(dict(quarter=t, quarter_start=starts,
        actual_mean_severity=np.round(sev,2),
        baseline_trend_fit=np.round(fitted,2),
        pct_above_baseline=np.round(dev*100,2),
        bocpd_alarm_prob=np.round(Ab,3),
        prism_alarm_prob=np.round(Ap,3)))
    claims.to_csv("results/wb_claims.csv",index=False)
    quarterly.to_csv("results/wb_quarterly.csv",index=False)
    trend.to_csv("results/wb_trend.csv",index=False)
    meta=dict(seed=int(seed), K=int(K), n_claims=int(len(claims)), claims_q0=int(BOOK.claims_q0),
              prism_first=int(fp) if fp else -1, bocpd_first=int(fb) if fb else -1,
              bocpd_max=float(Ab[4:].max()), baseline_slope_qtr=float(b1))
    import json; json.dump(meta, open("results/wb_meta.json","w"), indent=2)
    print("SEED",seed,"claims",len(claims),"PRISM fires q",fp,"BOCPD fires",fb,"BOCPD max",round(Ab[4:].max(),3))

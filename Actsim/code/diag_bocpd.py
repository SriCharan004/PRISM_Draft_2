import numpy as np, warnings
from dataclasses import replace
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import prism_sim as ps, prism_noext as px
cfg0=replace(ps.CFG, channel="severity"); K=cfg0.K
def act(a):
    i,scen=a; return px.panel_actsim(880000+i, scen, True, cfg0)
def actnone(i): return px.panel_actsim(905000+i,"gradual",False,cfg0)["logret"][1:]
if __name__=="__main__":
    with Pool(2) as p:
        act_grad=p.map(act,[(i,"gradual") for i in range(100)])
        act_abr =p.map(act,[(i,"abrupt")  for i in range(100)])
        actnoise=p.map(actnone, range(30))
    nat_grad=[ps.generate_panel(770000+i,"gradual",True,cfg0) for i in range(100)]
    nat_abr =[ps.generate_panel(770000+i,"abrupt", True,cfg0) for i in range(100)]
    sd_act=float(np.std(np.concatenate(actnoise)))
    sd_nat=float(np.std(np.concatenate([ps.generate_panel(905000+i,"gradual",False,cfg0)["logret"][1:] for i in range(30)])))
    cfg_act=replace(cfg0,sig_d=sd_act); cfg_nat=replace(cfg0,sig_d=sd_nat)
    def diag(panels,cfg,label):
        det=0; maxA=[]; ac=[]; pre=[]; post=[]
        for pp in panels:
            A=ps.bocpd(pp["logret"], cfg.h0, cfg); fa=ps.first_alarm(A,cfg)
            if fa is not None and K-2<=fa<=K+6: det+=1
            maxA.append(A[4:].max()); r=pp["logret"][1:]; ac.append(np.corrcoef(r[:-1],r[1:])[0,1])
            pre.append(np.mean(pp["logret"][5:K])); post.append(np.mean(pp["logret"][K+8:]))
        print(f"{label:16s} detect={det/len(panels):.3f}  meanMaxAlarm={np.mean(maxA):.3f}  ret_autocorr={np.mean(ac):+.3f}  step(post-pre)={np.mean(post)-np.mean(pre):+.4f}  vs sig_d={cfg.sig_d:.4f}",flush=True)
    print(f"sig_d: native={sd_nat:.4f} actsim={sd_act:.4f}\n")
    print("SEVERITY-ONLY BOCPD (alarm threshold 0.5):")
    diag(nat_grad,cfg_nat,"native GRADUAL"); diag(act_grad,cfg_act,"actsim GRADUAL")
    diag(nat_abr, cfg_nat,"native ABRUPT");  diag(act_abr, cfg_act,"actsim ABRUPT")
    # save one gradual example (native + actsim) alarm traces for a figure
    An=ps.bocpd(nat_grad[3]["logret"],cfg_nat.h0,cfg_nat); Aa=ps.bocpd(act_grad[3]["logret"],cfg_act.h0,cfg_act)
    np.savez("results/diag_bocpd.npz", nat_ret=nat_grad[3]["logret"], act_ret=act_grad[3]["logret"],
             An=An, Aa=Aa, K=K, sd_nat=sd_nat, sd_act=sd_act)
    print("[saved traces]")

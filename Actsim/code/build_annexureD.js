const fs=require('fs');
const d=require('docx');
const {Document,Packer,Paragraph,TextRun,HeadingLevel,Table,TableRow,TableCell,WidthType,ShadingType,BorderStyle,AlignmentType,ImageRun,PageBreak}=d;

const INK="1f2a37", BLUE="2E5A88", LIGHT="f6f8fa";
function P(text,opts={}){ // simple paragraph, text can be array of runs
  const runs = Array.isArray(text)? text : [new TextRun({text, size:21})];
  return new Paragraph({spacing:{after:120,line:276}, ...opts, children:runs});
}
function lead(boldPart, rest){ return P([new TextRun({text:boldPart, bold:true, size:21}), new TextRun({text:rest, size:21})]); }
function H1(text){ return new Paragraph({heading:HeadingLevel.HEADING_1, spacing:{before:240,after:120}, children:[new TextRun({text, bold:true, color:BLUE, size:26})]}); }
function H2(text){ return new Paragraph({heading:HeadingLevel.HEADING_2, spacing:{before:180,after:80}, children:[new TextRun({text, bold:true, color:INK, size:23})]}); }
function caption(text){ return new Paragraph({spacing:{before:60,after:180}, children:[new TextRun({text, italics:true, size:18, color:"5b6b7b"})]}); }

const TW=9026;
function cell(text,{w,head=false,bold=false,align="left"}={}){
  return new TableCell({
    width:{size:w,type:WidthType.DXA},
    shading:head?{type:ShadingType.CLEAR,fill:BLUE,color:"auto"}:{type:ShadingType.CLEAR,fill:"auto",color:"auto"},
    margins:{top:40,bottom:40,left:80,right:80},
    children:[new Paragraph({alignment:align==="left"?AlignmentType.LEFT:AlignmentType.CENTER,
      children:[new TextRun({text,bold:head||bold,color:head?"FFFFFF":INK,size:18})]})]
  });
}
function table(headers, rows, widths){
  const hr=new TableRow({tableHeader:true,children:headers.map((h,i)=>cell(h,{w:widths[i],head:true,align:i===0?"left":"center"}))});
  const trs=rows.map(r=>new TableRow({children:r.map((c,i)=>cell(String(c),{w:widths[i],align:i===0?"left":"center",bold:i===0}))}));
  return new Table({width:{size:TW,type:WidthType.DXA},columnWidths:widths,rows:[hr,...trs]});
}
function figure(path,widthPx,heightPx){
  return new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:120,after:60},
    children:[new ImageRun({type:"png",data:fs.readFileSync(path),transformation:{width:widthPx,height:heightPx}})]});
}

const kids=[];
kids.push(new Paragraph({spacing:{after:40},border:{bottom:{style:BorderStyle.SINGLE,size:12,color:BLUE}},children:[new TextRun({text:"Annexure D — Independent Claim-Level Validation on CAS ActSim",bold:true,size:30})]}));
kids.push(new Paragraph({spacing:{after:60},children:[new TextRun({text:"A robustness study of PRISM under an independent, claim-level data-generating process, with an outlier-robust adjustment and a characterisation across the curvature of a gradual inflection.",italics:true,size:20,color:"5b6b7b"})]}));
kids.push(new Paragraph({spacing:{after:200},children:[new TextRun({text:"Companion to PRISM (Draft 2 → Draft 3). Prepared by Dr. Rohan Yashraj Gupta (FIA, FIAI). 27 September 2026. All data synthetic.",size:18,color:"5b6b7b"})]}));

kids.push(H1("D.1 Motivation and relation to the main paper"));
kids.push(P("The Monte Carlo results of Section 4 draw the monitored severity series directly at the quarterly aggregate level, with Gaussian, independent log-returns — the detector's own observation model. That establishes correct behaviour under a known law but does not test the method under a data-generating process it does not assume. This annexure regenerates the loss side of every panel one claim at a time with CAS ActSim (Zhang; validated by the CAS Risk Working Group; github.com/casact/actsim) and aggregates the simulated claims into a quarterly mean-severity series, exactly as an actuary builds a trend exhibit from a claims listing. The data are therefore independent of the detector and carry realistic claim-level sampling noise."));
kids.push(lead("Scope. ","ActSim generates claims, not external economic indicators, so the external stream is unavailable here. The contrast tested in this annexure is therefore the loss-only benchmark (severity-only, constant-hazard BOCPD) against the internal-driven form of PRISM, extended with the outlier-robust adjustment introduced in D.3. The external stream — the paper's primary early-warning mechanism — is not exercised; its treatment is deferred to the roadmap in D.7. All data remain synthetic."));

kids.push(H1("D.2 Data: claim-level generation and calibration"));
kids.push(P("Each replication is a 40-quarter panel with inflection onset at quarter 24 and a 16-quarter frozen-baseline window, matching Section 3. The synthetic book carries approximately 4,000 claims per quarter across three severity classes with lognormal claim sizes; quarterly mean severity is the arithmetic mean of the simulated claims. Following the validation discipline of Section 3.1, the in-control series is confirmed against its target moments before any conclusion is drawn (Table D1)."));
kids.push(table(
  ["In-control metric","Target","ActSim","Verdict"],
  [["Mean severity log-return","0.00850","0.00844","within tolerance"],
   ["SD severity log-return","0.01600","0.01495","within tolerance"],
   ["KS test vs Normal, p-value","p > 0.05","0.317","cannot reject"],
   ["Lag-1 autocorrelation of log-return","≈ 0 (native −0.02)","−0.464","structural (see note)"]],
  [3200,1900,1900,2026]));
kids.push(caption("Table D1. Calibration of the ActSim panel (60 in-control replications, ~4,000 claims/quarter). The mean and dispersion match target, but the log-returns carry a strong negative lag-1 autocorrelation absent from the native process."));
kids.push(lead("Note on the autocorrelation. ","Under claim-level generation the sampling noise enters the level, not the return, so quarterly mean severity behaves as a mean plus independent sampling error rather than a random walk. Differencing then induces a lag-1 autocorrelation near −0.5. The Gaussian UPM assumes independent returns, so ActSim data is mildly misspecified for the detector — part of what makes it a fairer test."));

kids.push(H1("D.3 Method: the reversion-aware adjustment, PRISM(adjusted)"));
kids.push(P("A single anomalous quarter — one catastrophic-claim quarter that reverts the next quarter — is not a regime change, but constant-hazard BOCPD reads its large return as a changepoint. PRISM(adjusted) places a reversion-aware Hampel (median) filter ahead of the detector: each quarter is compared with its local median and an isolated spike is replaced, while a sustained shift, where several quarters sit at a new level, passes through unchanged. The internal predictor-stability signals are computed on the filtered series. The filter removes one-off outliers but preserves genuine inflections, at the cost of a one-to-two-quarter confirmation lag. Detection is reported as on-time — a first alarm within the window from two quarters before onset to six quarters after — with a late alarm not counted as a detection."));

kids.push(H1("D.4 Primary results: loss-only BOCPD vs PRISM(adjusted)"));
kids.push(P("Across 80 paired replications per regime, the pattern of Section 4 is reproduced under the independent generator (Table D2). Against an abrupt shock both monitors detect essentially every panel and the adjustment is immaterial. Against a gradual inflection the loss-only monitor is entirely blind (0 of 80), while PRISM(adjusted) detects a fifth of panels on time; the paired McNemar test is decisive. A weaker subtle inflection is caught less often but the direction is unchanged."));
kids.push(table(
  ["Regime (on-time detection)","BOCPD","PRISM(adjusted)","95% CI","McNemar"],
  [["Abrupt shock","1.000","0.988","[.933,.998]","tie (0 vs 1)"],
   ["Gradual inflection","0.000","0.200","[.127,.300]","16 vs 0, p=3.1×10⁻⁵"],
   ["Subtle inflection","0.000","0.075","[.035,.154]","6 vs 0, p=3.1×10⁻²"]],
  [3000,1400,1900,1500,1226]));
kids.push(caption("Table D2. On-time detection under ActSim (80 paired replications per regime, no external stream). McNemar counts are PRISM(adjusted)-only vs BOCPD-only discordant detections."));
kids.push(P("The two contributions are separable. A controlled decomposition (loss-only BOCPD, BOCPD with the filter, and PRISM(adjusted)) shows the filter alone cuts the one-off-outlier false-alarm rate from 1.00 to about 0.06 without lifting gradual detection, while the internal signal alone lifts gradual detection from zero. The false-alarm behaviour is reported in Table D4."));
kids.push(figure("results/fig_h2h_v3.png",600,247));
kids.push(caption("Figure D1. On-time detection on real inflections (left) and false-alarm rate on flat data and one-off outlier quarters (right), ActSim, n = 80 per scenario."));

kids.push(H1("D.5 Detectability across the curvature of a gradual inflection"));
kids.push(P("Holding the pre-inflection drift fixed and varying how steeply the growth rate climbs after onset traces a family of gradual inflections of increasing curvature. The loss-only monitor detects none of them at any steepness. PRISM(adjusted)'s on-time detection scales monotonically with the curvature, and its median lag shrinks as the bend sharpens (Table D3)."));
kids.push(table(
  ["Curvature","Post-shift growth","BOCPD on-time","PRISM(adjusted) on-time","Median lag (q)"],
  [["Gentle","5.1%/yr","0.000","0.025","12"],
   ["Mild","6.8%/yr","0.000","0.062","9"],
   ["Steep","8.5%/yr","0.000","0.162","7"],
   ["Very steep","10.2%/yr","0.000","0.400","7"]],
  [1900,2100,1900,2100,1026]));
kids.push(caption("Table D3. On-time detection and median detection lag across gradual inflections of increasing curvature (80 replications per curvature). BOCPD detects none at any curvature."));
kids.push(figure("results/fig_curve_v3.png",600,251));
kids.push(caption("Figure D2. On-time detection rising with steepness (left) and median detection lag shrinking with steepness (right)."));
kids.push(P("There is a soft detectability floor: a very gentle drift, only marginally above the baseline, is caught rarely (2.5%) and very late (12 quarters). Once the bend is moderately sharp, on-time detection climbs steadily to 40% at the steepest curvature, and the lag settles at about seven quarters."));

kids.push(H1("D.6 Outlier robustness (failure-mode check)"));
kids.push(P("Mirroring the failure-mode battery of Section 4.3, the key adversarial case under claim-level data is a one-off outlier quarter. Table D4 reports false-alarm rates on flat data and on transient spikes. The loss-only monitor false-alarms on every outlier; the reversion filter returns PRISM(adjusted) to near its flat-data rate."));
kids.push(table(
  ["Scenario (no real change)","BOCPD","PRISM(adjusted)","Reading"],
  [["Flat (in-control)","0.000","0.088","sensitivity cost"],
   ["One-off spike up","1.000","0.138","filter removes the spike"],
   ["One-off dip down","1.000","0.125","filter removes the dip"]],
  [2900,1500,1900,2726]));
kids.push(caption("Table D4. False-alarm rates under ActSim (80 replications per scenario; lower is better). The reversion filter cuts the outlier false-alarm rate from 1.00 to about 0.13, at the cost of an 8.8% false-alarm rate on genuinely flat data."));

kids.push(H1("D.7 Limitations and roadmap for reducing lag"));
kids.push(P("The data are synthetic; ActSim provides claim-level realism and independence from the detector, not real experience. The external stream is not exercised here, so this annexure speaks only to the internal-driven form of the method. The dominant remaining weakness is detection lag: even at the steepest gradual curvature, on-time detection is 40% and the median lag is seven quarters, so the method is timely confirmation rather than strict early warning under internal signals alone."));
kids.push(P("The lag decomposes into an information part, which is fundamental — a gradual change begins smaller than the quarter-to-quarter noise, so no loss-only monitor can flag it early without firing on noise — and an algorithm part, which is addressable, since the internal signal is a six-quarter rolling mean. In priority order: (i) the external leading indicator is the only mechanism that gets ahead of the change, because it carries information the loss series does not yet hold, and is the paper's primary early-warning claim; on ActSim it can be exercised by injecting a synthetic leading indicator and measuring the lead-time gain. (ii) A CUSUM signal in place of the rolling mean accumulates small persistent drifts and should pull detection earlier at matched false alarms — the correct version of the window-shortening instinct, which was tested and failed. (iii) Claim-level distributional monitoring exploits the tail and spread that aggregation to a single mean discards. (iv) Threshold lowering trades lead for false alarms along a frontier. (v) A larger book lowers the noise floor."));

kids.push(H1("D.8 Reproducibility"));
kids.push(P([new TextRun({text:"All results are reproducible from fixed seeds. Modules: ",size:21}),
  new TextRun({text:"prism_robust.py",font:"Consolas",size:19}),new TextRun({text:" (PRISM(adjusted), head-to-head, outliers), ",size:21}),
  new TextRun({text:"prism_curve.py",font:"Consolas",size:19}),new TextRun({text:" (curvature sweep); result files ",size:21}),
  new TextRun({text:"h2h_summary.json",font:"Consolas",size:19}),new TextRun({text:" and ",size:21}),
  new TextRun({text:"curve_summary.json",font:"Consolas",size:19}),new TextRun({text:". Loss data from CAS ActSim; the detector core prism_sim.py is imported unchanged, so no Section 4 number is affected. All data synthetic.",size:21})]));

const doc=new Document({
  styles:{default:{document:{run:{font:"Calibri",size:21,color:INK}}}},
  sections:[{properties:{page:{margin:{top:1440,bottom:1440,left:1440,right:1440}}},children:kids}]
});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync("PRISM_Draft3_AnnexureD.docx",b);console.log("written",b.length,"bytes");});

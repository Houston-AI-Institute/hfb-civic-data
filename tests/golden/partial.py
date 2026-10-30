import csv,sys,collections,pandas as pd,numpy as np
H=sys.argv[1]
C={'48201','48157','48039','48015','48473','48471','48071','48455','48407','48291'}
rows=[r for r in csv.DictReader(open(f'{H}/fit/zcta-county/rel.txt',encoding='utf-8-sig'),delimiter='|') if r['GEOID_ZCTA5_20']]
byz=collections.defaultdict(list)
for r in rows: byz[r['GEOID_ZCTA5_20']].append(r)
share={}
for z,rs in byz.items():
    ins=sum(int(r['AREALAND_PART']) for r in rs if r['GEOID_COUNTY_20'] in C)
    if ins>0: share[z]=ins/int(rs[0]['AREALAND_ZCTA5_20'])
elig=[z for z,s in share.items() if s>=.5]
print('touching',len(share),'>=50% land',len(elig))
a=pd.read_csv(f'{H}/gaps/acs_b05003/b05003.dat',sep='|',dtype=str)
a=a[a.GEO_ID.str.startswith('860Z200US')].copy(); a['Z']=a.GEO_ID.str[-5:]
for c in a.columns[1:-1]: a[c]=pd.to_numeric(a[c],errors='coerce')
a['adult']=a.B05003_E008+a.B05003_E019
a['cvap']=a.B05003_E009+a.B05003_E011+a.B05003_E020+a.B05003_E022
a=a.set_index('Z')
cmp=pd.read_csv(f'{H}/geography/out/compare_201_157_039_015_473_471_071_455_407_291.csv',dtype={'ZCTA':str}).set_index('ZCTA')
d=pd.DataFrame({'land':pd.Series(share)}).join(cmp[['vap_reg','vap_to','zcta_vap']]).join(a[['adult','cvap']])
d['in_vap_over_acs_adult']=d.zcta_vap/d.adult
d['to_cvap']=d.vap_to/d.cvap
e=d[d.land>=.5]
print('eligible with ACS:',e.cvap.notna().sum())
r=e.in_vap_over_acs_adult
print('in-county 2020 VAP / ACS adults (eligible): quantiles',r.quantile([.05,.1,.25,.5,.75,.9]).round(2).to_dict())
print('eligible ZCTAs with ratio <0.85:',(r<.85).sum(),' <0.7:',(r<.7).sum())
print(e[r<.85].sort_values('in_vap_over_acs_adult')[['land','zcta_vap','adult','in_vap_over_acs_adult','to_cvap']].round(3).to_string())
x=e[(e.cvap>=1000)]
print('turnout(2024)/CVAP among eligible, cvap>=1000: n',len(x),' quantiles',x.to_cvap.quantile([0,.05,.25,.5,.75,.95,1]).round(3).to_dict())
print('IQR in pts:',round(100*(x.to_cvap.quantile(.75)-x.to_cvap.quantile(.25)),1))
# rank sensitivity of to/cvap to area vs vap weights
x2=x.join(cmp[['ALAND20_to']])
x2['to_cvap_area']=x2.ALAND20_to/x2.cvap
print('Spearman to/CVAP vap-weights vs area-weights:',round(x2.to_cvap.rank().corr(x2.to_cvap_area.rank()),3))
pr=lambda s: s.rank(pct=True)*100
dd=(pr(x2.to_cvap)-pr(x2.to_cvap_area)).abs()
print('pct-rank shift |median|, p90, n>20pts:',round(dd.median(),1),round(dd.quantile(.9),1),(dd>20).sum())
# turnout/reg vs turnout/cvap rank agreement
x2['to_reg']=x2.vap_to/x2.vap_reg
print('Spearman to/reg vs to/CVAP:',round(x2.to_reg.rank().corr(x2.to_cvap.rank()),3))
d=d.join(a[['B05003_E001']])
e=d[d.land>=.5]
print('eligible with 0.5<=land<0.95 (straddle outer boundary):',((e.land<.95)).sum())
G=pd.read_csv(f'{H}/geography/out/blocks_201_157_039_015_473_471_071_455_407_291.csv',dtype={'GEOID20':str,'ZCTA':str}).merge(pd.read_csv(f'{H}/geography/out/block_gq_201_157_039_015_473_471_071_455_407_291.csv',dtype={'GEOID20':str}),on='GEOID20')
cs=G.groupby('ZCTA').corr_adult.sum()/G.groupby('ZCTA').vap.sum()
e=e.join(cs.rename('corrsh'))
rk=e[(e.B05003_E001>=1500)&(e.corrsh.fillna(0)<.10)]
print('rankable after land/pop/prison rules:',len(rk),' of which also flagged (reg<500 or cvap<1000 or in-VAP/ACS<0.85):',((rk.vap_reg<500)|(rk.cvap<1000)|(rk.in_vap_over_acs_adult<.85)).sum())

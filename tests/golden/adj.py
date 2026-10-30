import sys,pandas as pd,numpy as np
R=sys.argv[1]; tag='201_157_039_015_473_471_071_455_407_291'
B=pd.read_csv(f'{R}/out/blocks_{tag}.csv',dtype={'GEOID20':str,'ZCTA':str}).merge(pd.read_csv(f'{R}/out/block_gq_{tag}.csv',dtype={'GEOID20':str}),on='GEOID20')
B['ZCTA']=B.ZCTA.fillna('NONE'); B['vap_adj']=(B.vap-B.corr_adult).clip(lower=0)
vd=pd.read_csv(f'{R}/dl_tlc_vd24/vd24.csv')
def z(w):
    g=B.groupby(['VTDKEY','ZCTA'])[w].sum().reset_index(); t=g.groupby('VTDKEY')[w].transform('sum')
    g['s']=np.where(t>0,g[w]/t,np.nan); g=g.merge(vd,on='VTDKEY')
    return (g.s*g.Voter_Registration).groupby(g.ZCTA).sum()
a=z('vap'); b=z('vap_adj'); d=((b-a)/a*100).dropna()
print('reg count change, VAP vs VAP-minus-prison weights: median |%|',round(d.abs().median(),2),' ZCTAs >5%:',(d.abs()>5).sum()); print(d[d.abs()>5].round(1).to_dict())
zz=B.groupby('ZCTA')[['vap','vap_adj']].sum()
for k in ['77320','77002','77583','77535']: print(k,'reg/VAP',round(a[k]/zz.vap[k],2),'-> reg/adjVAP',round(b[k]/zz.vap_adj[k],2))

import sys,pandas as pd,numpy as np
H=sys.argv[1]; G=H+'/geography'; tag='201_157_039_015_473_471_071_455_407_291'
B=pd.read_csv(f'{G}/out/blocks_{tag}.csv',dtype={'GEOID20':str,'ZCTA':str}).merge(pd.read_csv(f'{G}/out/block_gq_{tag}.csv',dtype={'GEOID20':str}),on='GEOID20',how='left').fillna({'corr_adult':0})
B['ZCTA']=B.ZCTA.fillna(''); B['vap_adj']=(B.vap-B.corr_adult).clip(lower=0)
g=B.groupby(['VTDKEY','ZCTA'])[['vap_adj','ALAND20']].sum().reset_index()
t=g.groupby('VTDKEY').vap_adj.transform('sum'); ta=g.groupby('VTDKEY').ALAND20.transform('sum')
g['s']=np.where(t>0,g.vap_adj/t,g.ALAND20/ta)
T={}
for y in ['22','24']:
    vd=pd.read_csv(f'{G}/dl_tlc_vd{y}/vd{y}.csv').set_index('VTDKEY'); T['to'+y]=(g.s*g.VTDKEY.map(vd.Turnout)).groupby(g.ZCTA).sum(); T['reg'+y]=(g.s*g.VTDKEY.map(vd.Voter_Registration)).groupby(g.ZCTA).sum()
Z=pd.DataFrame(T); zb=B.groupby('ZCTA')[['total','vap','corr_adult','vap_adj']].sum(); Z=Z.join(zb)
# ACS B05003
rows={}
with open(f'{H}/gaps/acs_b05003/b05003.dat') as f:
    h=f.readline().rstrip('\n').split('|')
    for line in f:
        if line.startswith('860Z200US'):
            v=line.rstrip('\n').split('|'); z=v[0][-5:]
            if z in Z.index: rows[z]=dict(zip(h[1:],map(float,v[1:])))
A=pd.DataFrame(rows).T
A['acs_pop']=A.B05003_E001; A['cvap']=A.B05003_E009+A.B05003_E011+A.B05003_E020+A.B05003_E022
A['cvap_moe']=np.sqrt(A.B05003_M009**2+A.B05003_M011**2+A.B05003_M020**2+A.B05003_M022**2)
A['acs_vap']=A.B05003_E008+A.B05003_E019
Z=Z.join(A[['acs_pop','cvap','cvap_moe','acs_vap']])
Z=Z[Z.index!='']
Z['inreg_share']=Z.total/Z.acs_pop
Z['cvap_adj']=Z.cvap-Z.corr_adult
Z['P1']=(Z.to22+Z.to24)/(2*Z.cvap_adj); Z['P1_unadj']=(Z.to22+Z.to24)/(2*Z.cvap)
Z['cv_cvap']=Z.cvap_moe/1.645/Z.cvap
Z['reg_cvap']=Z.reg24/Z.cvap_adj
Z.to_csv(sys.argv[2])
print('ZCTAs',len(Z),' with ACS',Z.acs_pop.notna().sum())
print('in-region pop share (2020 block pop / ACS pop) distribution:',Z.inreg_share.describe(percentiles=[.05,.1,.25,.5]).round(3).to_dict())
lo=Z[(Z.inreg_share<0.9)&(Z.acs_pop>1500)].sort_values('inreg_share'); print('ZCTAs with ACS pop>1500 and <90% of pop in region:',len(lo)); print(lo[['total','acs_pop','inreg_share','P1']].round(3).to_string())
print('P1 describe',Z.P1.describe().round(3).to_dict())
print('P1>1:',(Z.P1>1).sum(),' P1<0 or cvap_adj<=0:',(Z.cvap_adj<=0).sum())
print(Z[(Z.P1>0.95)|(Z.cvap_adj<=0)][['total','acs_pop','cvap','corr_adult','cvap_adj','to24','P1']].round(2).to_string())
for k in ['77320','77002','77583','77535','77422','77340','77441','77493','77494']:
    if k in Z.index: r=Z.loc[k]; print(k,{c:round(r[c],3) for c in ['total','acs_pop','vap','acs_vap','cvap','corr_adult','cvap_adj','to24','P1_unadj','P1','reg_cvap','cv_cvap','inreg_share']})
print('CVAP CV>12%:',(Z.cv_cvap>.12).sum(),' >20%:',(Z.cv_cvap>.2).sum(),' >40%:',(Z.cv_cvap>.4).sum())

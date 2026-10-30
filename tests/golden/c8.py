import sys,pandas as pd,numpy as np
Z=pd.read_csv(sys.argv[1],dtype={"ZCTA":str}).set_index("ZCTA")
Z['corr_share']=Z.corr_adult/Z.vap
import csv
C={'48201','48157','48039','48015','48473','48471','48071','48455','48407','48291'}
r=pd.read_csv(sys.argv[3],sep='|',dtype=str,encoding='utf-8-sig'); r=r[r.GEOID_ZCTA5_20.notna()&(r.GEOID_ZCTA5_20!='')]; r['al']=r.AREALAND_PART.astype(float)
li=r.groupby('GEOID_ZCTA5_20').apply(lambda g:g[g.GEOID_COUNTY_20.isin(C)].al.sum()/max(g.al.sum(),1),include_groups=False)
Z['land_in']=li
E=Z[(Z.land_in>=.5)&(Z.acs_pop>=1500)&(Z.corr_share<.10)].copy()
print('eligible',len(E),' CVAP CV among eligible: median',round(E.cv_cvap.median(),3),' max',round(E.cv_cvap.max(),3),' >12%:',(E.cv_cvap>.12).sum())
rng=np.random.default_rng(1); N=2000
se=E.cvap_moe/1.645; num=(E.to22+E.to24)/2
P=np.empty((N,len(E)))
for i in range(N):
    cv=np.maximum(E.cvap.values+rng.normal(0,se.values),1)-E.corr_adult.values
    p=num.values/np.maximum(cv,1); P[i]=pd.Series(p).rank(pct=True).values*100
base=(num/E.cvap_adj).rank(pct=True)*100
lo=np.percentile(P,5,axis=0); hi=np.percentile(P,95,axis=0); w=hi-lo
print('90% percentile-rank interval width from CVAP sampling error only: median',round(np.median(w),1),' p90',round(np.percentile(w,90),1),' max',round(w.max(),1))
q=pd.cut(base,[0,20,40,60,80,100],labels=False)
qs=np.array([pd.cut(pd.Series(P[i]),[0,20,40,60,80,100],labels=False).values for i in range(N)])
chg=(qs!=q.values).mean(axis=0)
print('ZCTAs whose quintile flips in >25% of draws:',(chg>.25).sum(),'of',len(E))
bot=base.nsmallest(10).index; print('bottom-10 P1 ZCTAs (build-up candidates) with 90% rank interval:')
for k in bot: j=E.index.get_loc(k); print(' ',k,round(base[k],1),(round(lo[j],1),round(hi[j],1)),'cv',round(E.cv_cvap[k],3),'pop',int(E.acs_pop[k]),'inreg',round(E.inreg_share[k],2))

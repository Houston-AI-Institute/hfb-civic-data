# usage: build.py ROOT CNTYFIPS3 [CNTYFIPS3 ...]  -> builds block table, VTD->ZCTA weights, compares methods
import sys, os, time, pandas as pd, geopandas as gpd, pyogrio, numpy as np
R=sys.argv[1]; counties=sys.argv[2:]
t0=time.time()
pop=pd.read_csv(f'{R}/dl_tlc_blkpop/Blocks_Pop.txt',dtype={'FIPS':str,'CTBKEY':str})
pop=pop[pop.FIPS.isin(counties)].copy(); pop['GEOID20']='48'+pop.CTBKEY
vd=pd.read_csv(f'{R}/dl_tlc_vd24/vd24.csv',dtype={'CNTYVTD':str})
vtd_all=gpd.read_file(f'{R}/dl_tlc_vtds/VTDs_24PG.shp',engine='pyogrio')
out=[]
for c in counties:
    blk=gpd.read_file(f'{R}/dl_blk_{c}/tl_2020_48{c}_tabblock20.shp',engine='pyogrio',ignore_geometry=True)
    fc=pyogrio.read_dataframe(f'{R}/dl_faces_{c}/tl_2020_48{c}_faces.shp',columns=['STATEFP20','COUNTYFP20','TRACTCE20','BLOCKCE20','ZCTA5CE20'],read_geometry=False)
    fc['GEOID20']=fc.STATEFP20+fc.COUNTYFP20+fc.TRACTCE20+fc.BLOCKCE20
    fz=fc.groupby('GEOID20').ZCTA5CE20.agg(lambda s: '|'.join(sorted(set(x for x in s if isinstance(x,str) and x))))
    multi=(fz.str.contains(r'\|')).sum()
    pts=gpd.GeoDataFrame(blk[['GEOID20','ALAND20','TRACTCE20']],geometry=gpd.points_from_xy(blk.INTPTLON20.astype(float),blk.INTPTLAT20.astype(float)),crs='EPSG:4269').to_crs(vtd_all.crs)
    v=vtd_all[vtd_all.CNTY==int(c)][['CNTYVTD','VTDKEY','geometry']]
    j=gpd.sjoin(pts,v,how='left',predicate='within').drop(columns='index_right')
    j=j[~j.GEOID20.duplicated()]
    j['ZCTA']=j.GEOID20.map(fz).fillna('')
    j=j.merge(pop[['GEOID20','total','vap']],on='GEOID20',how='left')
    print(f'county {c}: blocks {len(blk)}, faces {len(fc)}, blocks w/ >1 ZCTA in faces {multi}, blocks no VTD {j.VTDKEY.isna().sum()} (pop {j.loc[j.VTDKEY.isna(),"total"].sum()}), blocks no ZCTA {(j.ZCTA=="").sum()} (pop {j.loc[j.ZCTA=="","total"].sum()}), pop matched {j.total.notna().sum()}/{len(j)} sum {j.total.sum():.0f}')
    out.append(j)
B=pd.concat(out); B['total']=B.total.fillna(0); B['vap']=B.vap.fillna(0)
B.drop(columns='geometry').to_csv(f'{R}/out/blocks_{"_".join(counties)}.csv',index=False)
# VTD-level weights
vd['VTDKEY']=vd.VTDKEY.astype(int)
def alloc(w):
    g=B.groupby(['VTDKEY','ZCTA'])[w].sum().reset_index()
    tot=g.groupby('VTDKEY')[w].transform('sum'); g['share']=np.where(tot>0,g[w]/tot,np.nan)
    # VTDs with zero weight: fall back to land area
    return g
res={}
for w in ['vap','total','ALAND20']:
    g=alloc(w)
    if w!='ALAND20':
        ga=alloc('ALAND20'); g=g.merge(ga[['VTDKEY','ZCTA','share']].rename(columns={'share':'sa'}),on=['VTDKEY','ZCTA'],how='outer')
        g['share']=g.share.where(g.share.notna(),g.sa)
    g=g.merge(vd[['VTDKEY','Voter_Registration','Turnout']],on='VTDKEY',how='left')
    g['reg']=g.share*g.Voter_Registration; g['to']=g.share*g.Turnout
    res[w]=g.groupby('ZCTA')[['reg','to']].sum()
    if w=='vap': g[['VTDKEY','ZCTA','share']].to_csv(f'{R}/out/vtd_zcta_weights_vap_{"_".join(counties)}.csv',index=False)
cmp=pd.concat({k:v for k,v in res.items()},axis=1)
cmp.columns=['_'.join(x) for x in cmp.columns]
cmp['zcta_vap']=B.groupby('ZCTA').vap.sum()
cmp['reg_diff_area_vs_vap_pct']=100*(cmp.ALAND20_reg-cmp.vap_reg)/cmp.vap_reg
cmp['rate_vap']=cmp.vap_to/cmp.vap_reg; cmp['rate_area']=cmp.ALAND20_to/cmp.ALAND20_reg
cmp['rate_diff_pts']=100*(cmp.rate_area-cmp.rate_vap)
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
print(cmp.round(3).sort_values('vap_reg',ascending=False).to_string())
vt=vd[vd.CNTYVTD.str[:3].isin(counties)]
print('TLC county totals reg/turnout:',vt.Voter_Registration.sum(),vt.Turnout.sum(),' allocated (vap):',round(cmp.vap_reg.sum()),round(cmp.vap_to.sum()),' VTDs in TLC file:',len(vt),' VTDs with blocks:',B.VTDKEY.nunique())
a=cmp[cmp.index!='']
print('median |reg diff| %:',a.reg_diff_area_vs_vap_pct.abs().median().round(1),' max:',a.reg_diff_area_vs_vap_pct.abs().max().round(1),' median |rate diff| pts:',a.rate_diff_pts.abs().median().round(2),' max:',a.rate_diff_pts.abs().max().round(2))
cmp.to_csv(f'{R}/out/compare_{"_".join(counties)}.csv')
print('secs',round(time.time()-t0,1))

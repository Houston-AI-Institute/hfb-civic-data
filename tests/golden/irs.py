import sys,csv,collections,pandas as pd
H=sys.argv[1]
C={'48201','48157','48039','48015','48473','48471','48071','48455','48407','48291'}
rows=[r for r in csv.DictReader(open(f'{H}/fit/zcta-county/rel.txt',encoding='utf-8-sig'),delimiter='|') if r['GEOID_ZCTA5_20']]
byz=collections.defaultdict(list)
for r in rows: byz[r['GEOID_ZCTA5_20']].append(r)
el=set(z for z,rs in byz.items() if sum(int(r['AREALAND_PART']) for r in rs if r['GEOID_COUNTY_20'] in C)/max(1,int(rs[0]['AREALAND_ZCTA5_20']))>=.5)
b=pd.read_csv(f'{H}/gaps/irs/eo_tx.csv',dtype=str,usecols=['ZIP','SUBSECTION','NTEE_CD','STATUS'])
b['Z']=b.ZIP.str[:5]
n=b.NTEE_CD.fillna('')
civ=(b.SUBSECTION=='04')|n.str.match(r'^(S2[012]|S8[012]|W24)')
x=b[civ & b.Z.isin(el)]
print('BMF rows',len(b),' civic rows in eligible ZCTAs',len(x),' of which 501c4',(x.SUBSECTION=='04').sum(), ' status values',b.STATUS.value_counts().head(3).to_dict())
cnt=x.groupby('Z').size().reindex(sorted(el),fill_value=0)
print('eligible ZCTAs',len(cnt),' zero',(cnt==0).sum(),' <=2',(cnt<=2).sum(),' median',cnt.median(),' max',cnt.max(), ' top5',cnt.sort_values().tail(5).to_dict())
print('blank NTEE share in eligible ZIP rows:', round((b[b.Z.isin(el)].NTEE_CD.fillna('')=='').mean(),3))
b2=pd.read_csv(f'{H}/gaps/irs/eo_tx.csv',dtype=str,usecols=['NAME','ZIP','SUBSECTION','NTEE_CD'])
b2['Z']=b2.ZIP.str[:5]; n2=b2.NTEE_CD.fillna('')
civ2=(b2.SUBSECTION=='04')|n2.str.match(r'^(S2[012]|S8[012]|W24)')
y=b2[civ2 & b2.Z.isin(el)]
hoa=y.NAME.str.contains(r'HOMEOWNER|HOME OWNER|PROPERTY OWNER|OWNERS ASS|COMMUNITY ASSOCIATION|MAINTENANCE|HOA\b|CONDOMINIUM|TOWNHOME|ESTATES|VILLAGE ASSOC',case=False,na=False)
print('civic rows',len(y),' HOA-like names',hoa.sum(),' share',round(hoa.mean(),3),' HOA-like among c4',(hoa&(y.SUBSECTION=='04')).sum())
print('NTEE of c4 rows (top):',y[y.SUBSECTION=='04'].NTEE_CD.fillna('blank').str[:3].value_counts().head(8).to_dict())
fix=y[~hoa & ~y.NTEE_CD.fillna('').str.startswith('L50')]
c=fix.groupby('Z').size().reindex(sorted(el),fill_value=0)
print('fixed I1 rows',len(fix),' zero',(c==0).sum(),' <=2',(c<=2).sum(),' median',c.median())

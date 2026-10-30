import sys,csv,collections,re,pandas as pd
H=sys.argv[1]; Z=pd.read_csv(sys.argv[2],dtype={'ZCTA':str}).set_index('ZCTA')
zs=set(Z.index)
civ=lambda code,sub: code[:3] in ("S20","S21","S22","S80","S81","S82","W24") or sub=="04"
hoa=re.compile(r'HOMEOWNER|HOME OWNER|PROPERTY OWNER|OWNERS ASSOC|MAINTENANCE|MUD\b|\bPOA\b|\bHOA\b|COMMUNITY ASSOC|CIVIC ASSOC|CIVIC CLUB|SUBDIVISION|ESTATES|LAKES|VILLAGE|TOWNHOME|CONDO')
hoa_strict=re.compile(r'HOMEOWNER|HOME OWNER|PROPERTY OWNER|OWNERS ASSOC|\bPOA\b|\bHOA\b|TOWNHOME|CONDO')
fire=re.compile(r'FIRE|EMS|RESCUE')
n=0; c=collections.Counter(); ntee4=collections.Counter(); byz=collections.Counter(); status=collections.Counter(); po=collections.Counter()
reg_prefix=('770','773','774','775','776','778','779')
cities=None
for row in csv.DictReader(open(f'{H}/gaps/irs/eo_tx.csv',encoding='latin-1')):
    z=(row['ZIP'] or '')[:5]; code=(row['NTEE_CD'] or '').strip(); sub=row['SUBSECTION']
    if not civ(code,sub): continue
    inz=z in zs
    if not inz:
        if z[:3] in ('770','772') : po['770/772 ZIP not a region ZCTA']+=1
        continue
    n+=1; nm=row['NAME'].upper()
    if sub=='04':
        c['c4']+=1; ntee4[code[:1] or 'blank']+=1
        if hoa_strict.search(nm): c['c4_name_hoa_strict']+=1
        elif hoa.search(nm): c['c4_name_hoa_loose']+=1
        if fire.search(nm): c['c4_name_fire_ems']+=1
    else: c['ntee_only']+=1
    if code=='' : c['blank_ntee']+=1
    byz[z]+=1
print('civic rows in region ZCTAs',n, dict(c)); print('c4 NTEE major group:',ntee4.most_common(12)); print(dict(po))
s=pd.Series(byz).reindex(Z.index).fillna(0)
el=Z[(Z.acs_pop>=1500)]
s=s.reindex(el.index)
print('eligible-ish (pop>=1500) ZCTAs',len(s),' count distribution:',s.describe().round(1).to_dict(),' zeros',(s==0).sum(),' <=2:',(s<=2).sum())
rate=s/el.acs_pop*1000
print('rate per 1000 describe',rate.describe().round(3).to_dict())
print('Spearman rate vs log pop',round(rate.rank().corr(el.acs_pop.rank()),3))

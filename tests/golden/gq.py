import sys,csv,pandas as pd
R=sys.argv[1]; tag='201_157_039_015_473_471_071_455_407_291'; cs=set(tag.split('_'))
lr={}
with open(f'{R}/dl_pl/txgeo2020.pl',encoding='latin-1') as f:
    for line in f:
        p=line.split('|')
        if p[2]=='750' and p[14] in cs: lr[p[7]]=p[9]
rows=[]
with open(f'{R}/dl_pl/tx000032020.pl',encoding='latin-1') as f:
    for line in f:
        p=line.rstrip('\n').split('|')
        if p[4] in lr: rows.append((lr[p[4]],int(p[5]),int(p[7]),int(p[12])))
g=pd.DataFrame(rows,columns=['GEOID20','gq','corr_adult','college'])
print('blocks',len(g),' GQ',g.gq.sum(),' adult correctional',g.corr_adult.sum(),' college housing',g.college.sum())
print('adult correctional by county:',g.groupby(g.GEOID20.str[2:5]).corr_adult.sum().to_dict())
B=pd.read_csv(f'{R}/out/blocks_{tag}.csv',dtype={'GEOID20':str,'ZCTA':str})
B=B.merge(g,on='GEOID20',how='left').fillna({'corr_adult':0,'college':0})
B['vap_adj']=(B.vap-B.corr_adult).clip(lower=0)  # corr_adult is all-age but adult facilities -> ~all 18+
z=B.groupby('ZCTA')[['vap','corr_adult','vap_adj']].sum(); z['corr_share']=z.corr_adult/z.vap
print('ZCTAs where adult correctional pop >5% of VAP:'); print(z[z.corr_share>.05].sort_values('corr_share',ascending=False).round(3).to_string())
B[['GEOID20','corr_adult','college']].to_csv(f'{R}/out/block_gq_{tag}.csv',index=False)

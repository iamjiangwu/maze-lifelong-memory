#!/usr/bin/env python3
"""Extended Data Figure 8: reward decoding and route PV controls.

Every build reads the included frozen source_data directory (no raw recordings).
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import json
import re
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
sys.dont_write_bytecode = True
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image

HERE = Path(__file__).resolve()
COL = dict(approach='#E29957', departure='#B96B2C', neutral='#C4C4C4', mid='#C4C4C4',
           rewarded='#E29957', timeout='#B96B2C', node2='#5B8E7D', node3='#9C8060',
           day1='#E29957', day2='#E29957', decoder='#E29957', home='#587785')
CANVAS = (7.5, 6.04)
PANELS = {
    'a': (.15,.10,2.15,1.95), 'b': (2.40,.10,2.75,1.95), 'c': (5.30,.10,2.05,1.95),
    'd': (.15,1.94,2.36,2.00),
    'e': (2.61,1.94,1.55,2.00), 'f': (4.26,1.94,1.55,2.00), 'i': (5.91,1.94,1.44,2.00),
    'g': (.15,3.81,1.90,2.08), 'h': (2.15,3.81,2.97,2.08), 'j': (5.22,3.81,2.13,2.08),
}
DISPLAY = dict(a='a',b='b',c='c',d='d',e='e',f='f',i='g',g='h',h='i',j='j')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m; spec.loader.exec_module(m); return m
def ci(v):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    assert len(v)>1
    m=float(v.mean()); h=float(stats.sem(v)*stats.t.ppf(.975,len(v)-1))
    return m,m-h,m+h
def exact(v):
    v=np.asarray(v,float); assert np.isfinite(v).all()
    signs=np.asarray(list(itertools.product([-1.,1.],repeat=len(v))))
    return float(np.mean(np.abs(signs@v)>=abs(v.sum())-1e-12))
def bh(p):
    p=np.asarray(p,float); order=np.argsort(p)
    q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out=np.empty_like(p); out[order]=np.minimum(q,1); return out
def stars(p): return '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'n.s.'
def configure():
    plt.rcParams.update({'font.family':'Arial','font.size':6.,'axes.labelsize':6.,'axes.titlesize':6.2,
        'xtick.labelsize':5.4,'ytick.labelsize':5.4,'legend.fontsize':4.9,'axes.linewidth':.6,
        'legend.frameon':False,
        'xtick.major.width':.5,'ytick.major.width':.5,'xtick.major.size':2.,'ytick.major.size':2.,
        'axes.grid':False,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'ed8-standardized-pv-v4',
        'savefig.facecolor':'white','savefig.bbox':None})

def freeze(out):
    """Verify the included numerical inputs before drawing any ED8 panel."""
    manifest = out / 'INPUT_MANIFEST.json'
    if not manifest.is_file():
        raise FileNotFoundError('ED8 requires the included inputs; pass --inputs-from SOURCE_DIR')
    rows = json.loads(manifest.read_text())['files']
    assert all(sha(out / row['frozen']) == row['sha256'] for row in rows), 'Frozen input drift'

def map_grid(frame):
    grid=np.full((15,15),np.nan)
    keep=frame.display_all_mice.astype(str).str.lower().eq('true')
    for row in frame[keep].itertuples(): grid[int(row.by),int(row.bx)]=row.display_decoding_error
    return grid

def read(out):
    sd=out/'source_data'
    data={k:pd.read_csv(sd/f'accepted_{k}.csv',dtype={'animal':str}) for k in ['h','c','routes','stats','b','maze_wall']}
    data['schematic_primitives']=load('ed8_frozen_schematic_primitives',sd/'accepted_schematic_primitives.py')
    pair=pd.read_csv(sd/'signal_comparison_primary_pairs.csv',dtype={'animal':str})
    num=pd.read_csv(sd/'signal_comparison_primary_numbers.csv')
    pair=pair[pair.signal.eq('S')&pair.norm.eq('normalized')].copy()
    num=num[num.signal.eq('S')&num.norm.eq('normalized')].copy()
    animal=pair.groupby(['panel','group','animal'],as_index=False).mean(numeric_only=True)
    animal['advantage']=animal.port-animal.away
    rep=pd.read_csv(sd/'replication_day2_pairs.csv',dtype={'animal':str})
    rep=rep[rep.signal.eq('S')&rep.comparison.eq('replication: Day i+1 vs Day i+91')]
    rep=rep.groupby(['group','animal'],as_index=False)[['port','away']].mean()
    rep['advantage']=rep.port-rep.away
    rn=pd.read_csv(sd/'replication_day2_numbers.csv')
    rn=rn[rn.signal.eq('S')&rn.comparison.eq('replication: Day i+1 vs Day i+91')]
    xp=pd.read_csv(sd/'strength_checks_profiles.csv',dtype={'animal':str})
    plotted_profiles=['same direction across gap: home to reward',
                      'same direction across gap: reward to home',
                      'cross direction across gap', 'cross direction across gap, shuffle']
    xp=xp[xp.measure.isin(plotted_profiles)]
    xp=xp.groupby(['measure','animal','bin'],as_index=False).pv.mean()
    xn=pd.read_csv(sd/'strength_checks_numbers.csv')
    xn=xn[xn.check.eq('cross-direction')&xn.item.eq('cross direction across gap')]
    return data,pair,num,animal,rep,rn,xp,xn

def verify(out,pair,num,animal,rep,rn,xp,xn):
    records=[]
    def one(panel,group,v,row,family):
        v=np.asarray(v,float); v=v[np.isfinite(v)]; m,lo,hi=ci(v); p=exact(v)
        expected=row['mean'] if 'mean' in row.index else row.advantage
        np.testing.assert_allclose([m,lo,hi,p],[expected,row.lo,row.hi,row.p_exact],rtol=0,atol=2e-13)
        assert len(v)==int(row.n_mice)
        records.append(dict(panel=panel,group=group,n_mice=len(v),mean=m,lo=lo,hi=hi,
                            p_exact=p,adjusted_p=row.q_BH,family=family))
    for row in num[num.p_exact.notna()].itertuples(index=False):
        row=pd.Series(row._asdict())
        if row['group']=='rewarded minus timeout':
            w=animal[animal.panel.eq('i')].pivot(index='animal',columns='group',values='advantage')
            one('i','rewarded minus timeout',w.rewarded-w.timeout,row,'single')
        elif row['item']=='port minus elsewhere' and row['family'] in ['fig4e','ed8efh']:
            g=animal[animal.panel.eq(row.panel)&animal.group.eq(row['group'])]
            one(row.panel,row['group'],g.advantage,row,str(row.family))
    for family in ['fig4e','ed8efh']:
        rows=[r for r in records if r['family']==family]
        if rows: np.testing.assert_allclose(bh([r['p_exact'] for r in rows]),[r['adjusted_p'] for r in rows],atol=2e-13,rtol=0)
    for _,row in rn.iterrows(): one('d_second_day',row['group'],rep[rep.group.eq(row['group'])].advantage,row,'second_day')
    rr=[r for r in records if r['panel']=='d_second_day']
    np.testing.assert_allclose(bh([r['p_exact'] for r in rr]),[r['adjusted_p'] for r in rr],atol=2e-13,rtol=0)
    measure='cross direction across gap'
    x=xp[xp.measure.eq(measure)].pivot(index='animal',columns='bin',values='pv')
    row=xn[xn.item.eq(measure)].iloc[0]
    one('j',measure,x[21]-x.loc[:,range(21)].mean(axis=1),row,'single')
    assert len(records)==14, 'Expected only the fourteen displayed route tests'
    displayed=animal[np.isfinite(animal[['port','away']]).all(axis=1)].copy()
    assert displayed.query("panel=='e'").groupby('group').size().to_dict()=={'home_to_reward':13,'reward_to_home':13}
    ref=pd.read_csv(out/'source_data/candidate_paired_display_summary.csv')
    summary=displayed[displayed.panel.isin(['e','f','h'])].groupby(['panel','group']).agg(n_mice=('animal','size'),away=('away','mean'),port=('port','mean')).reset_index()
    for r in summary.itertuples():
        v=ref[ref.panel.eq(r.panel)&ref.group.eq(r.group)].iloc[0]
        np.testing.assert_allclose([r.n_mice,r.away,r.port],[v.n_mice,v.first_mean,v.second_mean],atol=2e-13,rtol=0)
    animal.to_csv(out/'route_animal_values.csv',index=False)
    displayed.to_csv(out/'route_paired_display_values.csv',index=False)
    rep.to_csv(out/'replication_animal_values.csv',index=False)
    first=animal[animal.panel.eq('main')].pivot(index='animal',columns='group',values='advantage')
    order=first.mean(axis=1).sort_values().index
    second=rep.pivot(index='animal',columns='group',values='advantage').reindex(order)
    daypairs=first.reindex(order).add_prefix('day_i_vs_i90__').join(second.add_prefix('day_i1_vs_i91__'))
    daypairs.insert(0,'display_order_bottom_to_top',np.arange(len(daypairs)))
    daypairs.reset_index().to_csv(out/'animal_daypair_display.csv',index=False)
    xp.to_csv(out/'cross_direction_animal_profiles.csv',index=False)
    pd.DataFrame(records).to_csv(out/'route_statistics_verified.csv',index=False)
    return displayed,records

def figpanel(letter):
    fig=plt.figure(figsize=PANELS[letter][2:],facecolor='white')
    fig.text(.01,1-.12/PANELS[letter][3],DISPLAY[letter],fontsize=8,fontweight='bold',va='top')
    return fig
def axis(fig,l,b,w,h):
    fw,fh=fig.get_size_inches(); return fig.add_axes([l/fw,b/fh,w/fw,h/fh])
def clean(ax): ax.tick_params(direction='out',pad=1.2); ax.grid(False)
def bracket(ax,x0,x1,y,p,h=None):
    h=np.diff(ax.get_ylim())[0]*.023 if h is None else h
    ax.plot([x0,x0,x1,x1],[y-h,y,y,y-h],color='#202020',lw=.6,clip_on=False)
    ax.text((x0+x1)/2,y+h*.2,stars(p),ha='center',va='bottom',fontsize=5.4,fontweight='bold' if p<.05 else 'normal')
def draw_pair(ax,frame,a,b,positions,colors,markers=('s','o')):
    assert np.isfinite(frame[[a,b]]).all().all()
    for row in frame.itertuples(): ax.plot(positions,[getattr(row,a),getattr(row,b)],color='#CFCFCF',lw=.5,zorder=1)
    for x,col,color,marker in zip(positions,[a,b],colors,markers):
        ax.scatter(np.full(len(frame),x),frame[col],s=5,color=color,alpha=.45,lw=0,zorder=2)
        m,lo,hi=ci(frame[col]); ax.errorbar(x,m,yerr=[[m-lo],[hi-m]],fmt=marker,ms=4.2,color='black',
            mfc=color,mew=.6,lw=.8,capsize=1.6,zorder=3)

def drop(ax,x,y,size):
    vertices=np.asarray([(0,-1),(-.9,.05),(-.65,.9),(0,.9),(.65,.9),(.9,.05),(0,-1)])*size+[x,y]
    codes=[MplPath.MOVETO]+[MplPath.CURVE4]*6
    ax.add_patch(PathPatch(MplPath(vertices,codes),fc='#36A3FF',ec='none',clip_on=False,zorder=10))

def draw_accepted(data):
    figs={}
    fig=figs['a']=figpanel('a'); ax=axis(fig,.47,.25,1.51,1.51); cax=axis(fig,.25,.25,.065,1.51)
    cmap=plt.get_cmap('cividis').copy(); cmap.set_bad('white')
    im=ax.imshow(np.ma.masked_invalid(map_grid(data['h'])),origin='lower',extent=(-.5,14.5,-.5,14.5),
        interpolation='none',cmap=cmap,vmin=1,vmax=14,alpha=.95,zorder=1)
    wall=data['maze_wall']; ax.plot(wall.x,wall.y,color='black',lw=.85,solid_capstyle='projecting',zorder=3)
    for column,color,zorder in [('is_matched_control',COL['neutral'],7),('is_reward_proximal_proxy',COL['decoder'],8)]:
        for row in data['h'][data['h'][column].astype(str).str.lower().eq('true')].itertuples():
            ax.add_patch(Rectangle((row.bx-.55,row.by-.55),1.1,1.1,fill=False,ec='white',lw=2.05,zorder=zorder-.1))
            ax.add_patch(Rectangle((row.bx-.55,row.by-.55),1.1,1.1,fill=False,ec=color,lw=1.45,zorder=zorder))
    ax.set(xlim=(-.5,14.5),ylim=(14.5,-.5),aspect='equal'); ax.axis('off')
    reward=data['h'][data['h'].is_reward_proximal_proxy.astype(str).str.lower().eq('true')].iloc[0]
    drop(ax,15.25,float(reward.by),.42)
    cb=fig.colorbar(im,cax=cax); cb.set_ticks([2,6,10,14]); cb.set_label('Decoding error (bins)',fontsize=6.2,labelpad=1)
    cb.ax.tick_params(labelsize=5.3,width=.55,length=1.8,pad=1); cb.ax.yaxis.set_ticks_position('left'); cb.ax.yaxis.set_label_position('left'); cb.outline.set_linewidth(.55)
    fig=figs['b']=figpanel('b'); ax=axis(fig,.40,.30,2.25,1.35)
    order=[('d1v1','frame_balanced',0,'Day i\nequal frames'),('d1v1','visit_balanced',1,'Day i\nequal visits'),
           ('d2v1','frame_balanced',2.25,'Day i+1\nequal frames'),('d2v1','visit_balanced',3.25,'Day i+1\nequal visits')]
    st=data['stats'].set_index(['panel_role','subestimand'])
    for variant,method,x,_ in order:
        g=data['b'][data['b'].variant.eq(variant)&data['b'].resample_method.eq(method)]
        draw_pair(ax,g,'reward_proxy_error','matched_control_error',[x-.11,x+.11],[COL['decoder'],COL['neutral']],('o','s'))
        bracket(ax,x-.11,x+.11,10.25,float(st.loc[('B',f'{variant}:{method}'),'p_value']),.18)
    ax.set(xlim=(-.55,3.8),ylim=(0,13.1),ylabel='Decoding error (bins)')
    ax.set_xticks([r[2] for r in order],[r[3] for r in order]); clean(ax)
    fig.text((.40+2.25/2)/2.75,.98,'Uniform-prior decoding error',fontsize=6.7,fontweight='bold',va='top',ha='center')
    ax.legend(handles=[Line2D([],[],marker=m,ls='',mfc=c,mec='black',ms=4,label=t) for m,c,t in [('o',COL['decoder'],'Reward'),('s',COL['neutral'],'Matched control')]],
        loc='upper center',bbox_to_anchor=(.5,1.08),ncol=2,columnspacing=.75,handletextpad=.3)
    for tick in ax.get_xticklabels(): tick.set_horizontalalignment('center')
    fig=figs['c']=figpanel('c'); axes=[]
    for k,(variant,title,color) in enumerate([('d1v1','Day i',COL['day1']),('d2v1','Day i+1',COL['day2'])]):
        ax=axis(fig,.33+k*.84,.25,.72,1.15); axes.append(ax)
        g=data['c'][data['c'].pair_variant.eq(variant)]
        matched=g[['reward_target_to_reward_source_r','control_targets_to_own_control_source_mean_r']].to_numpy()
        cross=g[['reward_target_to_control_source_mean_r','control_targets_to_reward_source_mean_r']].to_numpy()
        for vals,filled in [(matched,True),(cross,False)]:
            color=COL['decoder'] if filled else COL['neutral']
            for row in vals: ax.plot([0,1],row,color=color,alpha=.20 if filled else .16,lw=.45,ls='-' if filled else '--',zorder=1)
            ax.scatter(np.tile([0,1],len(g)),vals.ravel(),s=6,color=color if filled else 'white',edgecolor='none' if filled else color,lw=.35,alpha=.28 if filled else .55,zorder=2)
            for x in (0,1):
                m,lo,hi=ci(vals[:,x]); ax.errorbar(x+(-.035 if filled else .035),m,yerr=[[m-lo],[hi-m]],fmt='o',ms=3.7,mfc=color if filled else 'white',mec=color,mew=.65,ecolor=color,lw=.65,capsize=1.4,zorder=5)
        ax.set_title(title,color='black',pad=2,fontweight='bold'); ax.set(xlim=(-.3,1.3),ylim=(-.1,.74))
        ax.set_xticks([0,1],['Reward\ntarget','Control\ntargets']); ax.tick_params(axis='x',labelsize=4.7)
        if not k: ax.set_ylabel('PV correlation',labelpad=1.5)
        else: ax.tick_params(labelleft=False)
        bracket(ax,-.035,1.035,.655,float(st.loc[('C',variant),'p_value']),.015); clean(ax)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=edge,mfc=fill,ms=3,label=lab) for edge,fill,lab in [(COL['decoder'],COL['decoder'],'Matched source'),(COL['neutral'],'white','Cross-location')]],
        loc='upper center',bbox_to_anchor=(.57,.91),ncol=1,fontsize=4.8,handletextpad=.3,labelspacing=.25,frameon=False)
    return figs

def draw_route(animal,rep,xp,records,data):
    figs={}; tests={(r['panel'],r['group']):r for r in records}
    dirs=[('home_to_reward',COL['approach'],'Home to reward'),('reward_to_home',COL['departure'],'Reward to home')]
    main=animal[animal.panel.eq('main')].pivot(index='animal',columns='group',values='advantage'); main=main.loc[main.mean(axis=1).sort_values().index]
    second=rep.pivot(index='animal',columns='group',values='advantage').reindex(main.index)
    assert main.notna().sum().to_dict()==second.notna().sum().to_dict()
    y=np.arange(len(main))
    fig=figs['d']=figpanel('d')
    for column,(frame,title) in enumerate([(main,'Day i → i + 90'),(second,'Day i + 1 → i + 91')]):
        ax=axis(fig,.38+column*1.02,.40,.82,1.20)
        for group,color,_ in dirs:
            mk,off=('D',.17) if group=='home_to_reward' else ('s',-.17)
            ax.scatter(frame[group],y+off,marker=mk,s=10,color=color,lw=0,zorder=2)
            ax.axvline(frame[group].mean(),color=color,lw=.8)
        ax.set_yticks(y,main.index)
        ax.set(ylim=(-.7,len(main)-.3),xlim=(-.13,.52),ylabel='Animal ID' if column==0 else '')
        ax.set_xticks([0,.2,.4]); clean(ax); ax.tick_params(labelsize=6.)
        ax.yaxis.label.set_fontsize(6.)
        if column: ax.tick_params(labelleft=False)
        ax.set_title(title,fontsize=6.,pad=2,fontweight='bold')
    fig.legend(handles=[Line2D([],[],marker=mk,ls='',color=c,mfc=c,ms=3.1,label=lab) for mk,c,lab in [
        ('D',COL['approach'],'Toward port'),('s',COL['departure'],'Away from port')]],
        loc='upper center',bbox_to_anchor=(.57,.99),ncol=2,fontsize=6.,handletextpad=.25,columnspacing=.6,borderaxespad=.2,frameon=False)
    fig.text(.55,.025,'Port − elsewhere\n(Standardized PV correlation)',ha='center',va='bottom',fontsize=6.)
    def block(fig,panel,groups,rect,ylabel,ylim,star_y,title):
        ax=axis(fig,*rect); ax.set(ylim=ylim,xlim=(-.65,3.65),ylabel=ylabel)
        for k,(group,color,label) in enumerate(groups):
            g=animal[animal.panel.eq(panel)&animal.group.eq(group)]
            draw_pair(ax,g,'away','port',[2*k,2*k+1],[COL['neutral'],color])
            bracket(ax,2*k,2*k+1,star_y,tests[panel,group]['adjusted_p'])
            ax.text(2*k+.5,-.18 if '\n' in label else -.23,label,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=4.7)
        ax.set_xticks([0,1,2,3],['Else-\nwhere','Reward','Else-\nwhere','Reward']); ax.tick_params(axis='x',labelsize=4.6)
        ax.set_title(title,fontsize=6.2,fontweight='bold',pad=4); ax.yaxis.label.set_fontsize(5.6); clean(ax)
        return ax
    short=[(g,c,'Toward\nreward' if g=='home_to_reward' else 'Away from\nreward') for g,c,_ in dirs]
    fig=figs['e']=figpanel('e'); block(fig,'e',short,(.40,.43,1.09,1.20),
        'Standardized PV correlation\n(across-gap / within-day)',(-.35,.85),.77,'Reliability-adjusted')
    fig=figs['f']=figpanel('f'); block(fig,'f',short,(.39,.43,1.10,1.20),'Standardized PV correlation',(-.12,.62),.56,'Equal frame control')
    fig=figs['g']=figpanel('g'); ax=axis(fig,.06,.25,1.78,1.58)
    primitives=data['schematic_primitives']
    for x,y,w,h in primitives.GRAY_BLOCKS:
        ax.add_patch(Rectangle((x,y),w,h,fc='#D9D9D9',ec='#D9D9D9',lw=.1,zorder=1))
    wall=data['maze_wall']; ax.plot(wall.x,wall.y,color='#111111',lw=.85,zorder=3)
    handles=[]
    for run,label,color in [('home_to_water','Home',COL['home']),('2_to_water','Node 2',COL['node2']),('3_to_water','Node 3',COL['node3'])]:
        g=data['routes'][data['routes'].run_label.eq(run)].sort_values('route_order'); x=g.display_x.to_numpy(); y=g.display_y.to_numpy()
        ax.plot(x,y,color=color,lw=1.55,solid_capstyle='round',zorder=6)
        ax.scatter(x[0],y[0],s=27,color=color,edgecolor='white',lw=.55,zorder=7)
        handles.append(Line2D([],[],color=color,lw=1.55,marker='o',mfc=color,mec='white',ms=5,label=label))
    primitives._draw_house_icon(ax,dict(xy=(-1.30,7.),size=.78,facecolor='#111111',door_color='white',zorder=9))
    primitives._draw_water_drop_icon(ax,dict(xy=(15.18,6.12),size=.75,facecolor='#36A3FF',edgecolor='#0866B2',highlight_color='#C9ECFF',linewidth=.75,zorder=9))
    ax.set(xlim=(-1.90,16.25),ylim=(15.35,-.75),aspect='equal'); ax.axis('off')
    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.50,-.02),ncol=3,columnspacing=.55,handlelength=1.4,handletextpad=.25,fontsize=4.8)
    fig=figs['h']=figpanel('h')
    for k,node in enumerate((2,3)):
        groups=[(f'node{node}_to_reward',COL['approach'],'Toward\nreward'),(f'reward_to_node{node}',COL['departure'],'Away from\nreward')]
        ax=block(fig,'h',groups,(.37+k*1.30,.40,1.14,1.30),'Standardized PV correlation' if k==0 else '',(-.12,.62),.56,f'Node {node}')
        ax.title.set_color(COL[f'node{node}'])
        if k: ax.tick_params(labelleft=False)
    fig=figs['i']=figpanel('i'); ax=axis(fig,.40,.43,.98,1.20)
    w=animal[animal.panel.eq('i')].pivot(index='animal',columns='group',values='advantage').reset_index()
    draw_pair(ax,w,'rewarded','timeout',[0,1],[COL['rewarded'],COL['timeout']],('o','s'))
    ax.set(xlim=(-.5,1.5),ylim=(-.3,.7),ylabel='Port − elsewhere\n(Standardized PV correlation)')
    ax.set_xticks([0,1],['Rewarded','Timeout']); ax.yaxis.label.set_fontsize(5.5); clean(ax)
    ax.set_title('Reward delivery',fontsize=6.2,fontweight='bold',pad=4)
    bracket(ax,0,1,.62,tests['i','rewarded minus timeout']['p_exact'])
    fig=figs['j']=figpanel('j'); ax=axis(fig,.38,.38,1.65,1.08)
    for measure,color,ls,lw,label in [('same direction across gap: home to reward',COL['approach'],'-',.9,'Toward vs toward'),
        ('same direction across gap: reward to home',COL['departure'],'--',.9,'Away vs away'),
        ('cross direction across gap','black','-',1.2,'Toward vs away'),('cross direction across gap, shuffle',COL['mid'],':',.9,'Neuron ID shuffle')]:
        g=xp[xp.measure.eq(measure)]; prof=np.asarray([ci(g[g.bin.eq(k)].pv) for k in range(22)])
        if measure=='cross direction across gap':
            ax.fill_between(range(22),prof[:,1],prof[:,2],color='black',alpha=.12,lw=0)
            cross_port_mean=prof[21,0]
        ax.plot(range(22),prof[:,0],color=color,ls=ls,lw=lw,label=label)
    ax.set(xlim=(-.5,22.5),ylim=(-.08,.45),ylabel='Standardized PV correlation',xlabel='Position along route')
    ax.set_xticks([0,21],['Home','Reward']); ax.yaxis.label.set_fontsize(5.6); clean(ax)
    fig.text(.57,.96,'Approach vs departure',fontsize=6.2,fontweight='bold',ha='center',va='top')
    ax.legend(loc='lower left',bbox_to_anchor=(-.02,1.00),ncol=2,fontsize=4.9,frameon=False,handlelength=1.2,columnspacing=.5,handletextpad=.35)
    ax.text(21.35,cross_port_mean,stars(tests['j','cross direction across gap']['p_exact']),fontsize=5.6,fontweight='bold',va='center')
    return figs

def bounds(fig,letter):
    fig.canvas.draw(); ren=fig.canvas.get_renderer(); failures=[]
    for t in fig.findobj(match=matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip(): continue
        b=t.get_window_extent(ren)
        if b.x0< -1 or b.y0< -1 or b.x1>fig.bbox.width+1 or b.y1>fig.bbox.height+1:
            failures.append(dict(panel=letter,text=t.get_text(),bbox=list(b.bounds)))
    return failures

def fit_ylabel_margins(fig):
    """Reserve extra left margin only if substituted fonts overflow the panel.

    Keep the original Arial geometry when it fits. Moving each axis's left edge
    by the same amount preserves paired-axis widths and fixed right boundaries;
    limits, plotted values, and font sizes are unchanged. Bounds are still tested
    on the final artwork below rather than accepting clipped labels.
    """
    adjustments=[]
    for _ in range(3):
        fig.canvas.draw(); renderer=fig.canvas.get_renderer()
        labels=[ax.yaxis.label for ax in fig.axes
                if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text()]
        left=min((label.get_window_extent(renderer).x0 for label in labels),default=0.)
        if left >= -1.: break
        shift=(1.-left)/fig.bbox.width
        for ax in fig.axes:
            pos=ax.get_position()
            if pos.width-shift <= .1:
                raise RuntimeError('Fallback font leaves insufficient room for the ED8 axis')
            ax.set_position([pos.x0+shift,pos.y0,pos.width-shift,pos.height])
        adjustments.append(dict(left_overflow_px=-left,axis_shift_fraction=shift))
    return adjustments
def compose(out):
    doc=fitz.open(); page=doc.new_page(width=CANVAS[0]*72,height=CANVAS[1]*72)
    ns='http://www.w3.org/2000/svg'; ET.register_namespace('',ns)
    root=ET.Element(f'{{{ns}}}svg',width=f'{CANVAS[0]*72}pt',height=f'{CANVAS[1]*72}pt',viewBox=f'0 0 {CANVAS[0]*72} {CANVAS[1]*72}')
    for letter,(x,y,w,h) in PANELS.items():
        p=out/'individual_panels'/f'ed8_panel_{DISPLAY[letter]}'
        with fitz.open(p.with_suffix('.pdf')) as src: page.show_pdf_page(fitz.Rect(x*72,y*72,(x+w)*72,(y+h)*72),src,0)
        xml=p.with_suffix('.svg').read_text(); tree=ET.fromstring(xml)
        ids={el.attrib['id']:f'{letter}_{el.attrib["id"]}' for el in tree.iter() if 'id' in el.attrib}
        for old,new in sorted(ids.items(),key=lambda kv:-len(kv[0])): xml=xml.replace(f'="{old}"',f'="{new}"').replace(f'#{old}',f'#{new}')
        tree=ET.fromstring(xml); group=ET.SubElement(root,f'{{{ns}}}g',transform=f'translate({x*72} {y*72})')
        for child in tree:
            if child.tag not in [f'{{{ns}}}metadata']: group.append(child)
    stem=out/'ed8_standardized_pv'; doc.set_metadata({'title':'Extended Data Figure 8: standardized route PV controls','creationDate':'','modDate':''})
    doc.save(stem.with_suffix('.pdf'),garbage=4,deflate=True); doc.close()
    ET.ElementTree(root).write(stem.with_suffix('.svg'),encoding='utf-8',xml_declaration=True)
    with fitz.open(stem.with_suffix('.pdf')) as pdf:
        for dpi in [300,600]:
            pix=pdf[0].get_pixmap(dpi=dpi,alpha=False); im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
            im.save(out/f'ed8_standardized_pv_{dpi}dpi.png',dpi=(dpi,dpi))
            if dpi==600: im.save(out/'ed8_standardized_pv_600dpi.tiff',dpi=(dpi,dpi),compression='tiff_lzw')

def documentation(out,pair,records):
    (out/'README.md').write_text('''# Extended Data Figure 8

Panels a–c show reward decoding and unstandardized maze PV correlation. Route panels use standardized S maps; panel c uses unstandardized C. Orange marks observed measurements and gray marks controls or shuffles. Direction is also distinguished by marker or line style. The route schematic shows home, water and the starting nodes. Visible labels use Reward; numerical source columns retain port.

Rows are a/b/c, d/e/f/g, and h/i/j. Panel d contains first-/second-day comparisons with identical animal order and axis limits. animal_daypair_display.csv contains both comparisons. PANEL_CROSSWALK.csv maps numerical analysis keys to display letters: i (reward delivery) is g, g (schematic) is h, and h (node controls) is i. Individual-panel filenames use display letters.

## Rebuild without raw recordings

From the repository root, run `python reproduce.py --figure ed8 --output-dir outputs/ed8` with a new output directory.

The default control/shuffle gray is #C4C4C4.

The builder reads the included CSV/JSON data and checks INPUT_MANIFEST.json. No finished composite or raw neural/behavior recording is a rebuild input. Software requirements are listed in the repository's reproduction guide.

## Validation and statistical scope

route_animal_values.csv includes the undefined departure reward ratio for mouse 434. route_paired_display_values.csv uses identical animals for both paired means: reliability panel e contains 13 mice in each direction. route_statistics_verified.csv recalculates route estimates, intervals, exact sign-flip tests and correction families. Panel d's second-day comparison and j's profiles are built from the included pair/bin rows. Panel b/c statistical annotations use the supplied result tables; the top-level statistics command independently replays their tests.

The route families use numerical source keys: Figure4e/d (two directions), replication/d (two directions), and e/f/i (eight contrasts). Display panel g's rewarded-minus-timeout and panel j's cross-direction contrasts are separate single tests. Error bars are 95% Student-t intervals across animals for route panels. Panel c uses pair-level points and intervals, with standard errors clustered by animal. Panel b uses one-sided exact sign-flip tests on animal means with BH4; route tests are two-sided.

## Files

PDF and SVG contain all ten panels; PNGs are 300 and 600 dpi and TIFF is 600 dpi LZW. Individual panels are in individual_panels. DISPLAY_VALIDATION.json and TEXT_BOUNDS.json record export and text-bound checks; they do not replace visual inspection.
''',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--inputs-from',type=Path,required=True)
    p.add_argument('--control-gray',choices=['dark','light'],default='light')
    args=p.parse_args(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    COL['neutral']=COL['mid']='#454545' if args.control_gray=='dark' else '#C4C4C4'
    if args.inputs_from:
        source=args.inputs_from.resolve()
        for name in ['source_data','INPUT_MANIFEST.json']:
            src=source/name; dest=out/name
            if src.is_dir(): shutil.copytree(src,dest,dirs_exist_ok=True)
            else: shutil.copy2(src,dest)
    freeze(out); configure(); data,pair,num,animal,rep,rn,xp,xn=read(out)
    displayed,records=verify(out,pair,num,animal,rep,rn,xp,xn)
    figs=draw_accepted(data); figs.update(draw_route(displayed,rep,xp,records,data))
    # Visible terminology follows Figure 4; numerical port columns stay intact.
    for fig in figs.values():
        for label in fig.findobj(match=matplotlib.text.Text):
            value=re.sub(r'\bPort\b','Reward',label.get_text())
            label.set_text(re.sub(r'\bport\b','reward',value))
    font_layout={letter:fit_ylabel_margins(fig) for letter,fig in figs.items()}
    da=figs['d'].axes
    assert len(da)==2 and da[0].get_xlim()==da[1].get_xlim() and da[0].get_ylim()==da[1].get_ylim()
    assert not any(t.get_visible() for t in da[1].get_yticklabels())
    assert len(figs['d'].legends)==1
    daypair_layout=dict(status='PASS',xlim=list(da[0].get_xlim()),ylim=list(da[0].get_ylim()),
        shared_mouse_order=[t.get_text() for t in da[0].get_yticklabels()],titles=[ax.get_title() for ax in da],
        common_xlabel=[t.get_text() for t in figs['d'].texts if 'Reward − elsewhere' in t.get_text()],
        legend_count=len(figs['d'].legends),right_visible_mouse_labels=0,
        axes_width_inches=[ax.get_position().width*PANELS['d'][2] for ax in da])
    (out/'DAYPAIR_LAYOUT_VALIDATION.json').write_text(json.dumps(daypair_layout,indent=2))
    pd.DataFrame([dict(panel=DISPLAY[k],source_panel=k,**dict(zip(['x','top','width','height'],v))) for k,v in PANELS.items()]).to_csv(out/'layout_inches.csv',index=False)
    roles=dict(a='accepted decoding map',b='accepted balanced decoding',c='accepted unstandardized maze PV',
               d='first-/second-day comparisons with shared axes and ordering',e='reliability-adjusted',
               f='equal frames',i='rewarded versus timeout',g='restored route schematic',h='node routes',j='cross direction')
    pd.DataFrame([dict(source_panel=k,display_panel=DISPLAY[k],role=roles[k]) for k in PANELS]).to_csv(out/'PANEL_CROSSWALK.csv',index=False)
    paneldir=out/'individual_panels'; paneldir.mkdir(exist_ok=True); failures=[]
    for letter in PANELS:
        fig=figs[letter]; failures+=bounds(fig,letter)
        for fmt in ['pdf','svg','png']: fig.savefig(paneldir/f'ed8_panel_{DISPLAY[letter]}.{fmt}',dpi=300,bbox_inches=None,transparent=fmt!='png')
        plt.close(fig)
    (out/'TEXT_BOUNDS.json').write_text(json.dumps(failures,indent=2))
    compose(out); documentation(out,pair,records)
    (out/'source_code').mkdir(exist_ok=True)
    if HERE!=(out/'source_code'/HERE.name): shutil.copy2(HERE,out/'source_code'/HERE.name)
    with fitz.open(out/'ed8_standardized_pv.pdf') as pdf:
        text=pdf[0].get_text(); assert 'Standardized PV correlation' in text and 'Reliability-adjusted' in text
        assert 'Near port fraction' not in text and 'Population vector correlation' not in text
        image_count=len(pdf[0].get_images()); pages=len(pdf)
    receipt=dict(status='RENDERED_AWAITING_VISUAL_QA' if not failures else 'TEXT_BOUNDS_FAILED',
        panel_count=len(PANELS),pdf_pages=pages,canvas_inches=CANVAS,native_font='Arial',route_rule_version='2026-09-27-uniform-port-grid-v4',
        route_statistic_checks=len(records),paired_reliability_mice={'toward':13,'away':13},reliability_display_panel=DISPLAY['e'],text_bounds_failures=failures,
        font_margin_adjustments=font_layout,control_gray=COL['mid'],palette=COL,source_sha256=sha(HERE),pdf_image_count=image_count,outputs={p.name:sha(p) for p in out.glob('ed8_standardized_pv*') if p.is_file()})
    (out/'DISPLAY_VALIDATION.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2)); assert not failures

if __name__=='__main__': main()

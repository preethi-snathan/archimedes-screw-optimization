"""Plots for the comparison study; raw failures remain in tables and rate plots."""
import csv
import json
from pathlib import Path
import plots  # configures the Matplotlib cache and headless backend
import matplotlib.pyplot as plt
import numpy as np
import constraints as C
import analysis

METHODS=['SLSQP','NelderMead','DiffEvolution']
LABELS=['SLSQP','Nelder–Mead','DE']
COLORS=['#0072B2','#D55E00','#009E73']


def read_rows(out):
    with (out/'results.csv').open() as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for k in ('obj','violation','model_calls','time','Ro','rho','P','alpha_deg'):
            r[k]=float(r[k])
        for k in ('N','run','nfev'):
            r[k]=int(r[k])
        for k in ('success','feasible'):
            r[k]=r[k]=='True'
    return rows


def finish(fig,path):
    fig.savefig(path,dpi=200,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def generate(out=Path('results/comparison'), dest=Path('figures/comparison')):
    out,dest=Path(out),Path(dest)
    dest.mkdir(parents=True,exist_ok=True)
    rows=read_rows(out)
    hist=json.loads((out/'histories.json').read_text())
    calls=json.loads((out/'history_model_calls.json').read_text())

    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for j,(method,label,color) in enumerate(zip(METHODS,LABELS,COLORS)):
        vals,rates=[],[]
        for N in range(1,5):
            rs=[r for r in rows if r['N']==N and r['method']==method]
            good=[r for r in rs if r['success'] and r['feasible']]
            vals.append(max(r['obj'] for r in good) if good else np.nan)
            rates.append(100*len(good)/len(rs))
        positions=np.arange(1,5)+(j-1)*.24
        axs[0].bar(positions,vals,width=.24,label=label,color=color)
        axs[1].bar(positions,rates,width=.24,label=label,color=color)
    for ax in axs:
        ax.set_xticks(range(1,5)); ax.set_xlabel('Flights N')
    axs[0].set_ylabel('Best accepted volume [m³/rev]')
    axs[1].set_ylabel('Converged and feasible runs [%]'); axs[1].set_ylim(0,105)
    axs[0].legend()
    fig.suptitle('Design capacity and solver acceptance')
    finish(fig,dest/'designs_and_success.png')

    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for method,label,color in zip(METHODS,LABELS,COLORS):
        rs=[r for r in rows if r['method']==method]
        for ax,key in zip(axs,('model_calls','time')):
            ax.plot(range(1,5),[np.mean([r[key] for r in rs if r['N']==N]) for N in range(1,5)],'o-',label=label,color=color)
            ax.set_yscale('log'); ax.set_xticks(range(1,5)); ax.set_xlabel('Flights N')
    axs[0].set_ylabel('Mean volume-model calls (all runs)')
    axs[1].set_ylabel('Mean solver runtime [s] (all runs)')
    axs[0].legend(); fig.suptitle('Computational cost under the recorded solver settings')
    finish(fig,dest/'computational_cost.png')

    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for N,ax in zip(range(1,5),axs.flat):
        for method,label,color in zip(METHODS,LABELS,COLORS):
            key=f'{method}_N{N}_run0'
            y=np.array([np.nan if v is None else -v for v in hist[key]])
            ax.step(calls[key],y,where='post',color=color,label=label)
        ax.set_xscale('log'); ax.set_title(f'N={N}'); ax.set_xlabel('Cumulative volume-model calls')
        ax.set_ylabel('Best feasible trial VT [m³/rev]')
    axs[0,0].legend()
    fig.suptitle('First run for each method (illustrative; not final-success certification)')
    finish(fig,dest/'convergence.png')

    fig,axs=plt.subplots(1,4,figsize=(12,4),layout='constrained')
    for N,ax in zip(range(1,5),axs):
        ref=max(r['obj'] for r in rows if r['N']==N and r['method']=='SLSQP' and r['success'] and r['feasible'])
        for j,(method,label,color) in enumerate(zip(METHODS,LABELS,COLORS)):
            rs=[r for r in rows if r['N']==N and r['method']==method]
            good=[r for r in rs if r['success'] and r['feasible']]
            offsets=np.linspace(-.12,.12,len(good))
            ax.scatter(j+offsets,[100*r['obj']/ref for r in good],s=16,color=color,alpha=.7)
            ax.text(j,1.01,f'{len(good)}/{len(rs)}',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
        ax.set_xticks(range(3),LABELS,rotation=25); ax.set_title(f'N={N}',pad=24)
        ax.axhline(100,color='gray',linestyle='--',linewidth=.7)
    axs[0].set_ylabel('Accepted VT / best SLSQP VT [%]')
    fig.suptitle('Variation across starts/seeds; labels show accepted/total runs')
    finish(fig,dest/'initial_guess_sensitivity.png')

    # A conditional slice, not a picture of the full four-dimensional region.
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    angles=np.linspace(20,40,81); pitches=np.linspace(.05,6,91)
    capacity=np.empty((len(pitches),len(angles)))
    for i,pitch in enumerate(pitches):
        for j,angle in enumerate(angles):
            x=np.array([1.,.53524,pitch,angle])
            v=C.volume_per_turn(x,4)
            capacity[i,j]=v if np.all(C.g_vector(x,4,volume=v)>=0) else np.nan
    im=ax.contourf(angles,pitches,np.ma.masked_invalid(capacity),levels=20,cmap='viridis')
    fig.colorbar(im,ax=ax,label='VT [m³/rev]')
    ax.axvline(np.degrees(np.arctan(C.H/C.L_MAX)),color='red',linestyle='--',label='Footprint boundary')
    best=max((r for r in rows if r['N']==4 and r['method']=='SLSQP' and r['success'] and r['feasible']),key=lambda r:r['obj'])
    ax.plot(best['alpha_deg'],best['P'],'r*',markersize=10,label='Best SLSQP design')
    ax.set_xlabel('Inclination [degrees]'); ax.set_ylabel('Pitch [m]')
    ax.set_title('Feasible slice: N=4, Ro=1 m, rho=0.53524\nWhite region is infeasible')
    ax.legend(fontsize=8)
    finish(fig,dest/'feasible_region_slice.png')

    evidence=json.loads((out/'convexity_evidence.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    t=np.linspace(0,1,81)
    for key,ax in zip(('objective','feasible_region'),axs):
        a=np.array(evidence[key]['points'][0]['x'])
        b=np.array(evidence[key]['points'][2]['x'])
        values=[analysis.evaluate_point((1-s)*a+s*b,4) for s in t]
        if key=='objective':
            y=[p['negative_VT'] for p in values]
            ax.plot(t,y,label='f = −VT',color=COLORS[0])
            ax.plot([0,1],[y[0],y[-1]],'--',label='Endpoint chord',color=COLORS[1])
            ax.set_ylabel('Objective f = −VT [m³/rev]')
            ax.set_title('Nonconvex objective: curve above chord')
        else:
            q=[C.Q_REQ*(p['residuals'][0]+1) for p in values]
            ax.plot(t,q,label='Flow along segment',color=COLORS[0])
            ax.axhline(C.Q_REQ,linestyle='--',color=COLORS[1],label='Minimum flow')
            ax.scatter([0,.5,1],[q[0],q[40],q[-1]],color=COLORS[0])
            ax.set_ylabel('Flow [m³/s]')
            ax.set_title('Nonconvex feasible region: midpoint fails')
        ax.set_xlabel('Position along segment, t'); ax.legend(fontsize=8)
    fig.suptitle('N=4 counterexamples on two different design segments')
    finish(fig,dest/'convexity_checks.png')


if __name__=='__main__':
    generate()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

from ggsc.io import read_h5ad_gs, save_fig_gs

from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf

import scanpy as sc
from scipy.stats import gmean


def mixedlmplot(x,y,pat_var,
                sample_var,
                data=None,
                ax=None,
                hue=None,
                palette=None,
                average_roi=True,
                **kwargs):
    
    if data is None:
        data = kwargs['data']
    
    if ax is None:
        ax = plt.gca()
        #f,ax = plt.subplots(1)
        
    md = smf.mixedlm(f"{y} ~ {x}",groups = 'patient',
            re_formula='1',vc_formula = {'sample_id' : '0 + C(sample_id)'},data = data)

    mdf = md.fit()
    p = mdf.pvalues[x]
    coef = mdf.fe_params
    
    
    # https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/j.2041-210x.2012.00261.x
    var_resid = mdf.scale
    var_random_effect = float(mdf.cov_re.iloc[0]) + float(mdf.vcomp)
    var_fixed_effect = mdf.predict(mdf.model.data.frame[x]).var()

    total_var = var_fixed_effect + var_random_effect + var_resid
    marginal_r2 = var_fixed_effect / total_var
    
    
    if average_roi:
        scatter_data = data.groupby(sample_var).mean(numeric_only=True)
        if hue is not None:
            scatter_data = scatter_data.join(data[[sample_var,hue]].drop_duplicates().set_index(sample_var))
    else:
        scatter_data = data
    g = sns.scatterplot(scatter_data,x=x,y=y,hue=hue,palette=palette)
    
    xl = np.array(ax.get_xlim())
    yval = xl * coef[x] + coef['Intercept']
    
    ax.plot(xl,yval,color='grey',linewidth=2)
    
    yl = ax.get_ylim()
    ax.text(xl[1],yl[0] + .04*(yl[1]-yl[0]),f'$R^2$={marginal_r2:.2f}\np={p:.2f}',ha='right')
    
    return(g)
    
def geomx_interaction_plot(adata_geo,
                source_gene,
                target_genes,
                source_region,
                target_region,
                hue=None,
                plot_type='regplot',
                average_roi=True,
                return_df = False,
                palette=None):

    def extract(adata,region,genes):
        df = adata[adata.obs['SegmentLabel']==region,genes].to_df()
        df['ROILabel'] = adata.obs['ROILabel']
        return(df)

    target_df = extract(adata_geo,target_region,target_genes)
    target_df = target_df.melt(id_vars = 'ROILabel',
                           var_name='target_gene',
                           value_name = 'expression')

    source_df = extract(adata_geo,source_region,source_gene)
    source_df['disease'] = adata_geo.obs['disease']
    source_df['sample_id'] = adata_geo.obs['sample_id']
    source_df['patient'] = adata_geo.obs['patient']
    
    df = pd.merge(source_df,target_df,on='ROILabel',how='inner')
    
    df = df[~df['patient'].isna()]
    
    xlabel = f'{source_gene} in {source_region}-rich region'
    ylabel = f'Expression in {target_region}-rich region'

    plot_df = df.rename(columns={source_gene : xlabel,'expression':ylabel})
    
    if plot_type=='regplot':
        g=sns.lmplot(data=plot_df,x=xlabel,y=ylabel,col='target_gene',col_wrap=3,
           facet_kws  = {'sharey': False},hue=hue)
    elif plot_type=='kde':
        g=sns.displot(data=plot_df,x=xlabel,y=ylabel,col='target_gene',col_wrap=3,
           facet_kws  = {'sharey': False},hue=hue,kind='kde',fill=True,alpha=.6,bw_adjust=1.25)
        g.map(sns.scatterplot)
    elif plot_type=='mixedlm':
        g = sns.FacetGrid(data=df,col='target_gene',col_wrap=3,sharex=False,sharey=False)
        g.map_dataframe(mixedlmplot,x=source_gene,y='expression',
                        pat_var='patient',
                        sample_var='sample_id',hue='disease',palette=palette,average_roi=average_roi)
        g.set_xlabels(xlabel,size=12)
        g.set_ylabels(ylabel,size=12)
    else:
        print('huh?')
        
    if return_df:
        return(g,df)
    else:
        return(g)
    

def draw_divider(pos,overhang,ax,color='k',linewidth=1):
    nytick = len(ax[0].get_yticks())
    ax[0].axhline(pos,xmin=-1*overhang,xmax=1,clip_on=False,color=color,linewidth=linewidth)
    ax[1].axhline(nytick - pos + .5,color=color,linewidth=linewidth)
    ax[2].axhline(pos,color=color,linewidth=linewidth)

def draw_subdivisions(ct_order,level_1,level_2,ax = None,color='grey'):

    i = 0
    for lab1,g1 in ct_order.groupby(level_1):
        draw_divider(i,overhang=2,ax=ax,linewidth=2,color=color)
        ax[0].text(-2,i,lab1,va='bottom',fontsize=12)
    
        for lab2,g2 in g1.groupby(level_2):
            draw_divider(i,overhang=1,ax=ax,linewidth=1,color=color)
            ax[0].text(-.1,i+.1,lab2,va='top',ha='right',fontsize=10)
    
            i+=g2.shape[0]
    ax[0].set_yticks([])
            
def make_lr_plot(X,ligands,receptors,top_n=10,linewidth=2,cbar=False):
    
    def extract(X,genes,linewidth=5):
        if type(genes) is dict:
            res = list()
            for c,gene_list in genes.items():
                s = pd.Series(gmean(X[gene_list],axis=1),index=X.index)
                s.name=c
                res.append(s)
            R = pd.concat(res,axis=1)
        else:
            R = X[genes]
        return(R)
    
    # Calculate gene or complex expression
    df1 = extract(X,ligands)
    df2 = extract(X,receptors)
    
    # Calculate interaction scores
    max_score = np.max((df1.values.reshape(-1,1,df1.shape[1],1) * df2.values.reshape(1,-1,1,df2.shape[1])),axis=(2,3))
    I = pd.DataFrame(max_score,index=df1.index,columns=df2.index)
    
    top_I = I.reset_index().rename(columns={'index':'L'}).\
        melt(id_vars='L',var_name='R',value_name='score').\
        sort_values('score',ascending=False).head(top_n)
    
    
    ## Do the plotting
    f,ax = plt.subplots(1,3,gridspec_kw={'width_ratios' : [df1.shape[1],3,df2.shape[1]]},
                        figsize=(5,6))


    # Ligand and receptor heatmaps
    g1=sns.heatmap(df1,cmap="Reds",ax=ax[0],cbar=False)
    g2=sns.heatmap(df2,cmap="Reds",ax=ax[2],cbar=False)
    
    # Manually add colorbars
    if cbar:
        # Define new axes for colorbars
        cbar_ax1 = f.add_axes([-.15, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar_ax2 = f.add_axes([1.2, 0.15, 0.02, 0.7])

        # Plot heatmaps again (invisibly) to extract colorbar info
        sns.heatmap(df1, cmap="Reds", cbar=True, cbar_ax=cbar_ax1, ax=ax[0],cbar_kws={'label' : 'Ligand expression (TPM)'})
        sns.heatmap(df2, cmap="Reds", cbar=True, cbar_ax=cbar_ax2, ax=ax[2],cbar_kws={'label' : 'Receptor expression (TPM)'})
        
        # Move ticks to left side for left colorbar
        cbar_ax1.yaxis.set_ticks_position('left')
        cbar_ax1.yaxis.set_label_position('left')



    # Interactions line
    for ind,row in top_I.iterrows():
        L_pos = df1.shape[0] - list(df1.index).index(row['L'])
        R_pos = df2.shape[0] - list(df2.index).index(row['R'])
        ax[2].tick_params(right=True,labelright=True,left=False,labelleft=False,labelrotation=0)
    
    
        score_pct = row['score'] / top_I.iloc[0]['score']
    
        ax[1].plot([0,1],[L_pos,R_pos],linewidth=linewidth*score_pct,color='k',alpha=score_pct * .9)
    
        ax[1].set_ylim([.5,df1.shape[0]+.5])
        for side in ['left','right','top','bottom']:
            ax[1].spines[side].set_visible(False)
        ax[1].set_yticks([])
        ax[1].set_xticks([])
    
    plt.subplots_adjust(wspace=0, hspace=0)
    
        
    
    return(ax)
#%%
# =============================================================================
# setores_b3.py
#
# Distribuição por SETOR B3 do universo investável, conforme a classificação
# usada em simulationFund.py (rule='SECTOR' — confirmado).
#
# Gera figures/setores/pie_sector (PDF + PNG):
#   - donut por setor, SEM título;
#   - rótulos elegantes junto a cada fatia (setor, nº de empresas, % do universo);
#   - contagem total de tickers investíveis evidenciada no centro;
#   - legenda (caption) descritiva no rodapé.
#
# Base: tickers negociáveis de datasets/BR_equity_closing_prices.csv mapeados
# por prefixo a REDUCED_TICKER (mesma regra de getLabels em simulationFund.py).
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SECT_DIR = 'paper/figures/setores'
os.makedirs(SECT_DIR, exist_ok=True)

LEVELS = ['SECTOR', 'SUB_SECTOR', 'SEGMENT']

# tradução dos setores B3 -> inglês (paper internacional)
SECTOR_EN = {
    'Petróleo, Gás e Biocombustíveis': 'Oil, Gas & Biofuels',
    'Materiais Básicos':               'Basic Materials',
    'Bens Industriais':                'Industrial Goods',
    'Consumo não Cíclico':             'Consumer Staples',
    'Consumo Cíclico':                 'Consumer Discretionary',
    'Saúde':                           'Health Care',
    'Tecnologia da Informação':        'Information Technology',
    'Comunicações':                    'Communications',
    'Utilidade Pública':               'Utilities',
    'Financeiro':                      'Financials',
    'Outros':                          'Others',
    'Unknown':                         'Unknown',
}

plt.rcParams.update({
    'figure.dpi':       110,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'font.size':        10,
    'font.family':      'serif',
    'mathtext.fontset': 'cm',
})


def investable_labels():
    """Mesma regra de simulationFund.getLabels (rule='SECTOR')."""
    cls = pd.read_excel('datasets/SectoralBrazilianClassification.xlsx', index_col=0)
    cols = pd.read_csv('datasets/BR_equity_closing_prices.csv', index_col=0, nrows=1).columns
    tickers = [c for c in cols if c not in ('IBOV', 'SELIC')]
    lab = pd.DataFrame(index=tickers, columns=LEVELS, dtype=object)
    for red in cls.REDUCED_TICKER.unique():
        fill = [t for t in tickers if t.startswith(red)]
        if fill:
            row = cls.loc[cls.REDUCED_TICKER == red, LEVELS].iloc[0]
            for c in LEVELS:
                lab.loc[fill, c] = row[c]
    return lab.fillna('Unknown')


labels = investable_labels()
sector_order = labels.SECTOR.value_counts().index.tolist()
_palette = list(plt.get_cmap('tab10').colors) + list(plt.get_cmap('Set2').colors)
SECTOR_BASE = {s: _palette[i] for i, s in enumerate(sector_order)}

print(f'Universo investável: {len(labels)} tickers | {labels.SECTOR.nunique()} setores B3.')


#%%
def plot_sector_pie(save=True, show=False):
    counts = labels.SECTOR.value_counts()                 # ordem decrescente
    total = int(counts.sum())

    # ordem de desenho: maiores no topo (flanqueando o 12h), menores embaixo.
    # pares (1º,3º,5º...) descem pelo lado direito; ímpares sobem pelo esquerdo.
    desc = counts.index.tolist()
    plot_order = desc[0::2] + desc[1::2][::-1]
    n = counts.reindex(plot_order).values
    colors = [SECTOR_BASE[s] for s in plot_order]

    fig, ax = plt.subplots(figsize=(9.0, 6.8))
    wedges, _ = ax.pie(n, colors=colors, startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor='white', linewidth=1.3))
    ax.set_aspect('equal')

    # contagem total no centro (evidencia o tamanho do universo investível)
    ax.text(0, 0.13, f'{total}', ha='center', va='center',
            fontsize=30, fontweight='bold', color='0.15')
    ax.text(0, -0.17, 'investable tickers\n(B3 universe)', ha='center', va='center',
            fontsize=9.5, color='0.35', linespacing=1.25)

    # legenda única à direita: setor, nº de empresas e % do universo (ordem decrescente)
    handles = [Patch(facecolor=SECTOR_BASE[s], edgecolor='white',
                     label=f'{SECTOR_EN.get(s, s)}  —  {counts[s]} companies · {counts[s] / total * 100:.1f}%')
               for s in counts.index]
    leg = ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5),
                    frameon=False, fontsize=9.5, handlelength=1.1, handleheight=1.1,
                    labelspacing=0.7, title='B3 sector', title_fontsize=10.5,
                    alignment='left')
    leg.get_title().set_fontweight('bold')

    # legenda (caption) descritiva — sem título no topo
    fig.text(0.5, 0.015,
             f'Sector composition of the B3 investable universe used in the fund simulation '
             f'({total} tradable tickers, B3 SECTOR classification): number of listed companies '
             f'and share of the universe per sector.',
             ha='center', va='top', fontsize=8, style='italic', color='0.35', wrap=True)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    if save:
        base = os.path.join(SECT_DIR, 'pie_sector')
        fig.savefig(base + '.pdf'); fig.savefig(base + '.png')
    plt.show() if show else plt.close(fig)


plot_sector_pie(save=True, show=False)
print(f'Figura salva em "{SECT_DIR}/pie_sector".')
# %%

#%%
# =============================================================================
# analise_experimentos.py
#
# Organização dos resultados para o paper.
#
# 1) Extrai as séries temporais (<Experimento>_Ts_WL_*) -> retornos diários por
#    estratégia, com e sem custo.
# 2) Plota as séries temporais (retorno acumulado) e SALVA em figures/.
#    - 1 figura por Window Length;
#    - série SEM custo (linha cheia) e COM custo (tracejada) no mesmo eixo;
#    - cor = estratégia (paleta colorblind-friendly), legendas enxutas.
# 3) Estende a extração para TODAS as medidas por carteira (<Experimento>_Ms_WL_*).
# 4) Analisa as medidas por carteira e tece comentários.
#
# Convenções de nomenclatura do paper:
#   SOHRP     ->  S-OHRP
#   SOHRPvol  ->  S-OHRP (Vol)  ->  subscrito Vol (ex.: S-OHRP_Vol)
# =============================================================================
import os
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde, linregress, wilcoxon, ttest_rel, t as _student_t
from Measures import Measures

# -----------------------------------------------------------------------------
# Experimento a analisar (prefixos gerados por simulationFund.py).
# >>> ESCOLHA AQUI: 1 (ExperimentoSOHRP, base do paper) ou 2 (ExperimentoOHRP) <<<
# As figuras/tabelas são gravadas nas pastas COMPARTILHADAS paper/figures e
# paper/tables; analise um experimento por vez (re-executar sobrescreve as saídas).
# -----------------------------------------------------------------------------
# 'wls'      = todos os WL carregados (séries, medidas e TABELAS usam todos);
# 'grid_wls' = subconjunto usado nas FIGURAS EM GRADE (1 painel por WL) para encolher
#              o grid; None => usa todos os 'wls'.
EXPERIMENTS = {
    1: {'name': 'ExperimentoSOHRP',
        'wls': [round(0.2 * i, 2) for i in range(1, 11)],   # 0.20 .. 2.00
        'grid_wls': None},                                  # grade usa os 10 WL
    2: {'name': 'ExperimentoOHRP',
        'wls': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        'grid_wls': [1.0, 2.0, 3.0, 4.0, 5.0]},             # grade só com anos inteiros
}
EXPERIMENT_TO_RUN = 2

EXP_PREFIX = EXPERIMENTS[EXPERIMENT_TO_RUN]['name']
wls = EXPERIMENTS[EXPERIMENT_TO_RUN]['wls']
_grid = EXPERIMENTS[EXPERIMENT_TO_RUN].get('grid_wls')
GRID_WLS = wls if _grid is None else [w for w in wls if w in _grid]
print(f'Analisando {EXP_PREFIX} ({len(wls)} window lengths; '
      f'{len(GRID_WLS)} nas figuras em grade).')

FIG_DIR = 'paper/figures/series_temporais'
DIST_DIR = 'paper/figures/distribuicoes'
BAR_DIR = 'paper/figures/barras_anuais'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DIST_DIR, exist_ok=True)
os.makedirs(BAR_DIR, exist_ok=True)

selic = pd.read_csv('datasets/SELIC_returns.csv', index_col=0, parse_dates=True)


# -----------------------------------------------------------------------------
# Nomenclatura / estética
# -----------------------------------------------------------------------------
def _rename_strategy(name):
    """SOHRP -> S-OHRP; SOHRPvol -> S-OHRP (Vol) (mantém ' (Vol)' se já existir)."""
    if name == 'SOHRPvol':
        return 'S-OHRP (Vol)'
    return name.replace('SOHRP', 'S-OHRP')


def disp(name):
    """Rótulo para exibição: ' (Vol)' -> subscrito Vol (mathtext)."""
    if name.endswith(' (Vol)'):
        return name[:-6] + r'$_{\mathrm{Vol}}$'
    return name


# paleta colorblind-friendly (Okabe-Ito); S-OHRP e sua variante Vol no mesmo
# tom de azul para comunicar parentesco.
STRAT_COLORS = {
    'S-OHRP':       '#0072B2',
    'S-OHRP (Vol)': '#56B4E9',
    'OHRP':         '#D55E00',
    'HRP':          '#009E73',
    'RP':           '#CC79A7',
    'EW':           '#E69F00',
}
SELIC_COLOR = '#222222'
STRAT_ORDER = ['S-OHRP', 'S-OHRP (Vol)', 'OHRP', 'HRP', 'RP', 'EW']
# grupo em destaque; demais estratégias entram esmaecidas
HIGHLIGHT = {'S-OHRP', 'S-OHRP (Vol)', 'OHRP'}

# períodos de estresse (sombreados/hachurados) — datação CODACE
CRISIS_SPANS = [
    ('2014-04-01', '2016-12-31', 'Brazilian recession\n(2014–2016)', '////', 0.62),
    ('2020-02-01', '2020-06-30', 'COVID-19\ncrisis (2020)',          'xxxx', 0.975),
]

plt.rcParams.update({
    'figure.figsize':   (7.2, 4.3),
    'figure.dpi':       110,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'font.size':        10,
    'font.family':      'serif',
    'mathtext.fontset': 'cm',
    'axes.titlesize':   11,
    'axes.titleweight': 'bold',
    'axes.labelsize':   10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linewidth':    0.5,
    'legend.fontsize':   8.5,
    'legend.frameon':    False,
})


# =============================================================================
# 1) EXTRAÇÃO DAS SÉRIES TEMPORAIS
# =============================================================================
# results[wl] = {'returns': df, 'returns_cost': df}
results = {}


def _load_ts_frame(path, suffix=''):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    cost_cols   = [c for c in df.columns if c.startswith('Returns with Cost ')]
    return_cols = [c for c in df.columns if c.startswith('Returns ') and c not in cost_cols]

    def _ret(c):  return _rename_strategy(c.removeprefix('Returns ')) + suffix
    def _cost(c): return _rename_strategy(c.removeprefix('Returns with Cost ')) + suffix

    returns      = df[return_cols].rename(columns=_ret)
    returns_cost = df[cost_cols].rename(columns=_cost)
    return returns, returns_cost


for wl in wls:
    path = f'results/{EXP_PREFIX}_Ts_WL_{wl:.2f}.csv'
    try:
        returns, returns_cost = _load_ts_frame(path)

        shared_dates = returns.index.intersection(selic.index)

        results[wl] = {
            'returns':      returns.loc[shared_dates].join(selic),
            'returns_cost': returns_cost.loc[shared_dates].join(selic),
        }
    except FileNotFoundError:
        print(f'[missing] {path}')
    except Exception as e:
        print(f'[error] {path} -> {e}')


#%%
# =============================================================================
# 2) PLOT DAS SÉRIES TEMPORAIS  (retorno acumulado, base 100) + SALVAR
# =============================================================================
def _cum_index(r):
    """Retorno acumulado normalizado para iniciar em 100."""
    r = r.dropna()
    g = (1 + r).cumprod()
    return g / g.iloc[0] * 100.0


def plot_timeseries(wl, returns, returns_cost, save=True, show=False):
    strat_cols = [c for c in STRAT_ORDER if c in returns.columns]

    fig, ax = plt.subplots()

    SELIC_LS = (0, (6, 1.5, 1, 1.5))  # dash-dot longo, distinto das estratégias

    # períodos de estresse: hachurados e identificados (atrás das linhas)
    span_trans = ax.get_xaxis_transform()  # x em dados, y em eixo
    for x0, x1, label, hatch, ytext in CRISIS_SPANS:
        a, b = pd.Timestamp(x0), pd.Timestamp(x1)
        ax.axvspan(a, b, facecolor='0.55', alpha=0.10, hatch=hatch,
                   edgecolor='0.55', linewidth=0.0, zorder=0)
        ax.text(a + (b - a) / 2, ytext, label, transform=span_trans,
                ha='center', va='top', fontsize=7.5, color='0.25', linespacing=1.2,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7',
                          lw=0.5, alpha=0.85), zorder=6)

    # somente NET OF COSTS. cor = estratégia; todas na MESMA opacidade/espessura.
    for c in strat_cols:
        if c not in returns_cost.columns:
            continue
        color = STRAT_COLORS.get(c, '#444444')
        gc = _cum_index(returns_cost[c])
        ax.plot(gc.index, gc.values, color=color, lw=1.5, ls='-', alpha=1.0, zorder=3)

    # SELIC como referência risk-free (preto, dash-dot)
    if 'SELIC' in returns.columns:
        s = _cum_index(returns['SELIC'])
        ax.plot(s.index, s.values, color=SELIC_COLOR, lw=1.4, ls=SELIC_LS, zorder=5)

    # eixos
    ax.set_ylabel('Cumulative return, net of costs (base 100)')
    ax.set_xlabel('')
    ax.set_yscale('log')
    ax.set_ylim(60, 1000)
    ax.set_yticks([50, 100, 200, 400, 800])
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.margins(x=0.01)

    # legenda de estratégias (cor) — todas na mesma opacidade
    strat_handles = [Line2D([0], [0], color=STRAT_COLORS[c], lw=2.0, label=disp(c))
                     for c in strat_cols if c in returns_cost.columns]
    if 'SELIC' in returns.columns:
        strat_handles.append(Line2D([0], [0], color=SELIC_COLOR, lw=1.4,
                                    ls=SELIC_LS, label='SELIC'))
    ax.legend(handles=strat_handles, loc='upper left',
              ncol=2, columnspacing=1.1, handlelength=1.8,
              title='Strategy', title_fontsize=8.5)

    # descrição (antigo título) no canto inferior direito
    ax.text(0.985, 0.04,
            f'Cumulative return by strategy\nWindow Length (WL) = {wl:.2f} years',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, fontweight='bold', linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.8', lw=0.6, alpha=0.9))

    fig.tight_layout()
    if save:
        base = os.path.join(FIG_DIR, f'retorno_acumulado_WL_{wl:.2f}_{EXP_PREFIX}')
        fig.savefig(base + '.pdf')
        fig.savefig(base + '.png')
    if show:
        plt.show()
    else:
        plt.close(fig)


for wl in sorted(results.keys()):
    plot_timeseries(wl, results[wl]['returns'], results[wl]['returns_cost'],
                    save=True, show=False)

print(f'\n{len(results)} figuras de série temporal salvas em "{FIG_DIR}/" (PDF + PNG).')


#%%
# =============================================================================
# 3) MEDIDAS POR CARTEIRA  (arquivos "Ms")
# =============================================================================
MEASURE_COLS = [
    'Sharpe Ratio', 'Sortino Ratio', 'Standard Deviation', 'Mean Return',
    'Drawdown', 'Pain Index', 'Sectoral Gini Index', 'Gini Index',
    'Turnover Ratio',
]


def _load_ms_frame(path, wl):
    df = pd.read_csv(path, parse_dates=['EntryDateRef', 'SortieDateRef'])
    df['Strategy'] = df['Strategy'].map(_rename_strategy)
    df['WL'] = wl
    return df


_frames = []
for wl in wls:
    path = f'results/{EXP_PREFIX}_Ms_WL_{wl:.2f}.csv'
    try:
        _frames.append(_load_ms_frame(path, wl))
    except FileNotFoundError:
        print(f'[missing] {path}')

measures = pd.concat(_frames, ignore_index=True)
measures = measures.sort_values(['WL', 'Strategy', 'SortieDateRef']).reset_index(drop=True)

measures_by_wl = {}
for wl, g in measures.groupby('WL'):
    measures_by_wl[wl] = g.set_index(['Strategy', 'SortieDateRef'])[MEASURE_COLS]

print(f'\nMedidas por carteira: {len(measures)} carteiras '
      f'({measures.Strategy.nunique()} estratégias x {len(wls)} WL).')
print('Estratégias:', sorted(measures.Strategy.unique()))


#%%
# =============================================================================
# 4) ANÁLISE DAS MEDIDAS POR CARTEIRA
# =============================================================================
def summary_by_strategy(measures, agg='mean'):
    return measures.groupby('Strategy')[MEASURE_COLS].agg(agg)


def measure_vs_wl(measures, measure, agg='mean'):
    return measures.pivot_table(index='WL', columns='Strategy',
                                values=measure, aggfunc=agg)


print('\n===== Média das medidas por carteira (pooled em todos os WL) =====')
print(summary_by_strategy(measures, 'mean').round(4).to_string())
print('\n===== Mediana das medidas por carteira =====')
print(summary_by_strategy(measures, 'median').round(4).to_string())

consistency = measures.assign(pos=measures['Sharpe Ratio'] > 0).groupby('Strategy').agg(
    sharpe_mean=('Sharpe Ratio', 'mean'),
    sharpe_std=('Sharpe Ratio', 'std'),
    sharpe_pos_frac=('pos', 'mean'),
)
print('\n===== Consistência do Sharpe por carteira =====')
print(consistency.round(3).to_string())


#%%
# --- gráficos das medidas (cores consistentes com as séries temporais) -------
def _order_present(measures):
    base = STRAT_ORDER + [s for s in sorted(measures.Strategy.unique())
                          if s not in STRAT_ORDER]
    return [s for s in base if s in set(measures.Strategy)]


def plot_strategy_bars(measures, measure, agg='mean'):
    order = _order_present(measures)
    s = summary_by_strategy(measures, agg)[measure].reindex(order)
    colors = [STRAT_COLORS.get(c, '#444444') for c in order]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar([disp(c) for c in order], s.values, color=colors, edgecolor='white')
    ax.set_title(f'{measure} — per-portfolio {agg} (all WL)')
    ax.set_ylabel(measure); ax.axhline(0, color='black', lw=0.6)
    plt.xticks(rotation=0); fig.tight_layout(); plt.show()


def plot_measure_vs_wl(measures, measure, agg='mean'):
    piv = measure_vs_wl(measures, measure, agg).reindex(columns=_order_present(measures))
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for c in piv.columns:
        ax.plot(piv.index, piv[c].values, marker='o', ms=4,
                color=STRAT_COLORS.get(c, '#444444'), label=disp(c))
    ax.set_title(f'{measure} (per-portfolio {agg}) vs window length')
    ax.set_xlabel('Window length'); ax.set_ylabel(measure)
    ax.legend(loc='best', ncol=2, title='Strategy', title_fontsize=8.5)
    fig.tight_layout(); plt.show()


def plot_measure_box(measures, measure):
    order = _order_present(measures)
    data = [measures.loc[measures.Strategy == s, measure].dropna() for s in order]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    bp = ax.boxplot(data, tick_labels=[disp(c) for c in order], showmeans=True,
                    patch_artist=True)
    for patch, c in zip(bp['boxes'], order):
        patch.set_facecolor(STRAT_COLORS.get(c, '#cccccc')); patch.set_alpha(0.55)
    ax.set_title(f'Per-portfolio distribution of {measure}')
    ax.set_ylabel(measure); ax.axhline(0, color='black', lw=0.6)
    plt.xticks(rotation=0); fig.tight_layout(); plt.show()


for measure in ['Sharpe Ratio', 'Turnover Ratio', 'Standard Deviation',
                'Drawdown', 'Gini Index', 'Sectoral Gini Index']:
    plot_strategy_bars(measures, measure, 'mean')

for measure in ['Sharpe Ratio', 'Turnover Ratio', 'Standard Deviation', 'Drawdown']:
    plot_measure_vs_wl(measures, measure, 'mean')

plot_measure_box(measures, 'Sharpe Ratio')
plot_measure_box(measures, 'Turnover Ratio')


#%%
# =============================================================================
# 4b) DISTRIBUIÇÕES COMPARATIVAS POR MÉTRICA
#     1 figura por métrica; 1 subplot por WL empilhado (A), (B), ...; em cada
#     painel, densidades (KDE) das estratégias sobrepostas para comparação.
# =============================================================================
_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def _robust_range(vals, clip):
    """Janela X = núcleo de maior densidade (recorta caudas p/ não achatar picos)."""
    lo, hi = np.percentile(vals, clip)
    if hi - lo < 1e-12:                 # quase-constante
        lo, hi = vals.min() - 0.5, vals.max() + 0.5
    return lo, hi


def _draw_tail_rug(ax, x, lo, hi, color):
    """Observações fora do núcleo: ticks fixados na borda (massa de cauda)."""
    trans = ax.get_xaxis_transform()    # x em dados, y em fração do eixo
    for tail, edge in [(x[x < lo], lo), (x[x > hi], hi)]:
        if len(tail):
            ys = np.linspace(0.015, 0.11, len(tail))
            ax.plot(np.full(len(tail), edge), ys, transform=trans, ls='none',
                    marker='|', ms=5, mew=1.0, color=color, alpha=0.55,
                    clip_on=False, zorder=4)


def plot_metric_distributions(measures, metric, clip=(2, 98), ncols=2,
                              save=True, show=False):
    measures = measures[measures.WL.isin(GRID_WLS)]      # grade só com os WL escolhidos
    wls_present = sorted(measures.WL.unique())
    order = _order_present(measures)
    n = len(wls_present)
    nrows = int(np.ceil(n / ncols))

    # eixo X robusto e compartilhado: foco no núcleo; caudas vão para o rug
    vals = measures[metric].dropna()
    lo, hi = _robust_range(vals, clip)
    pad = (hi - lo) * 0.04
    grid = np.linspace(lo, hi, 400)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 1.45 * nrows + 1.0),
                             sharex=True, layout='constrained')
    axes = np.atleast_1d(axes).ravel()
    has_tail = False

    for i, (wl, ax) in enumerate(zip(wls_present, axes)):
        sub = measures[measures.WL == wl]
        for s in order:
            x = sub.loc[sub.Strategy == s, metric].dropna()
            color = STRAT_COLORS.get(s, '#444444')
            if len(x) >= 5 and x.std() > 1e-9:
                d = gaussian_kde(x)(grid)
                ax.plot(grid, d, color=color, lw=1.3, zorder=3)
                ax.fill_between(grid, d, color=color, alpha=0.04, zorder=2)
            elif len(x) > 0:
                ax.axvline(np.clip(x.mean(), lo, hi), color=color, lw=1.4, zorder=3)
            if len(x) and ((x < lo).any() or (x > hi).any()):
                _draw_tail_rug(ax, x, lo, hi, color)
                has_tail = True
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.margins(y=0.10)
        ax.grid(axis='x', alpha=0.25)
        # eixo X: 5 valores em TODOS os painéis (mesmo com sharex) e fonte reduzida
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='x', labelbottom=True, labelsize=6)
        ax.text(0.025, 0.90, f'({_LETTERS[i]})  WL = {wl:.2f} yr',
                transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')

    for ax in axes[n:]:                 # remove painéis sobrando (se houver)
        ax.set_visible(False)

    fig.supxlabel(f'{metric}  (per portfolio)', fontsize=9)
    fig.supylabel('Density', fontsize=9)

    handles = [Line2D([0], [0], color=STRAT_COLORS[s], lw=2.2, label=disp(s))
               for s in order]
    leg = fig.legend(handles=handles, loc='outside upper center', ncol=len(order),
                     frameon=False, fontsize=8.5, handlelength=1.6, columnspacing=1.2,
                     title=f'Distribution of {metric} across individual portfolios, by strategy',
                     title_fontsize=9)
    leg.get_title().set_fontweight('bold')
    if has_tail:
        fig.text(0.5, -0.005, 'Edge ticks (|) mark out-of-range observations (distribution tails).',
                 ha='center', va='top', fontsize=7, color='0.4', style='italic')

    if save:
        slug = metric.lower().replace(' ', '_')
        base = os.path.join(DIST_DIR, f'dist_{slug}_{EXP_PREFIX}')
        fig.savefig(base + '.pdf'); fig.savefig(base + '.png')
    if show:
        plt.show()
    else:
        plt.close(fig)


DIST_METRICS = ['Sectoral Gini Index', 'Gini Index', 'Sharpe Ratio',
                'Turnover Ratio', 'Standard Deviation', 'Drawdown',
                'Sortino Ratio', 'Pain Index', 'Mean Return']
for metric in DIST_METRICS:
    plot_metric_distributions(measures, metric, save=True, show=False)

print(f'\n{len(DIST_METRICS)} figuras de distribuição salvas em "{DIST_DIR}/" (PDF + PNG).')


#%%
# =============================================================================
# 4c) BARRAS ANUAIS POR ESTRATÉGIA
#     Evolução da métrica no tempo (média das carteiras por ano), barras
#     agrupadas por estratégia; 1 subplot por WL (grade 5x2), (A)..(J).
#     Métricas anualizadas quando cabível (Std/Mean/Sharpe/Sortino, vide Measures.py).
# =============================================================================
# rótulo do eixo Y: marca as métricas já anualizadas na origem
_ANNUALIZED = {'Standard Deviation', 'Mean Return', 'Sharpe Ratio', 'Sortino Ratio'}

# faixas de crise em ANOS (eixo categórico das barras)
_BAR_CRISES = [((2014, 2016), '////'), ((2020, 2020), 'xxxx')]


def _shade_crises_bars(ax, years):
    """Sombreia faixas de crise mapeando intervalos de ano -> posições das barras."""
    for (y0, y1), hatch in _BAR_CRISES:
        idx = [k for k, y in enumerate(years) if y0 <= y <= y1]
        if idx:
            ax.axvspan(min(idx) - 0.5, max(idx) + 0.5, facecolor='0.55', alpha=0.13,
                       hatch=hatch, edgecolor='0.6', linewidth=0.0, zorder=0)


def plot_metric_bars_by_year(measures, metric, ncols=2, save=True, show=False):
    measures = measures[measures.WL.isin(GRID_WLS)]      # grade só com os WL escolhidos
    order = _order_present(measures)
    wls_present = sorted(measures.WL.unique())
    m = measures.assign(Year=measures['SortieDateRef'].dt.year)
    years = sorted(m['Year'].unique())
    x = np.arange(len(years))
    width = 0.8 / len(order)

    n = len(wls_present)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 1.9 * nrows + 1.0),
                             sharex=True, sharey=True, layout='constrained')
    axes = np.atleast_1d(axes).ravel()

    for i, (wl, ax) in enumerate(zip(wls_present, axes)):
        _shade_crises_bars(ax, years)
        piv = (m[m.WL == wl].groupby(['Year', 'Strategy'])[metric].mean()
               .unstack().reindex(index=years, columns=order))
        for j, s in enumerate(order):
            off = (j - (len(order) - 1) / 2) * width
            ax.bar(x + off, piv[s].values, width, color=STRAT_COLORS.get(s, '#444444'),
                   label=disp(s), linewidth=0, zorder=2)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=90, fontsize=6)
        ax.text(0.012, 0.95, f'({_LETTERS[i]})  WL = {wl:.2f} yr',
                transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')

    for ax in axes[n:]:
        ax.set_visible(False)

    ylab = metric + (' (annualized)' if metric in _ANNUALIZED else '')
    fig.supxlabel('Year (portfolio sortie)', fontsize=9)
    fig.supylabel(f'{ylab} — per-portfolio yearly mean', fontsize=9)

    handles = [Line2D([0], [0], color=STRAT_COLORS[s], lw=6, label=disp(s))
               for s in order]
    leg = fig.legend(handles=handles, loc='outside upper center', ncol=len(order),
                     frameon=False, fontsize=8.5, handlelength=1.2, columnspacing=1.2,
                     title=f'{ylab} per individual portfolio, averaged by year and strategy',
                     title_fontsize=9)
    leg.get_title().set_fontweight('bold')

    fig.text(0.5, -0.005,
             'Shaded: Brazilian recession (2014–2016, ////) and COVID-19 crisis (2020, xxxx).',
             ha='center', va='top', fontsize=7, color='0.4', style='italic')

    if save:
        slug = metric.lower().replace(' ', '_')
        base = os.path.join(BAR_DIR, f'bars_{slug}_{EXP_PREFIX}')
        fig.savefig(base + '.pdf'); fig.savefig(base + '.png')
    if show:
        plt.show()
    else:
        plt.close(fig)


BAR_METRICS = ['Standard Deviation', 'Mean Return', 'Sharpe Ratio',
               'Turnover Ratio', 'Drawdown', 'Pain Index',
               'Sortino Ratio', 'Gini Index', 'Sectoral Gini Index']
for metric in BAR_METRICS:
    plot_metric_bars_by_year(measures, metric, save=True, show=False)

print(f'\n{len(BAR_METRICS)} figuras de barras anuais salvas em "{BAR_DIR}/" (PDF + PNG).')


#%%
# =============================================================================
# 4d) TABELA: MODELAGEM DAS MÉTRICAS NO TEMPO (regressão linear por ano)
#     Para cada (WL, estratégia, métrica): regressão da média ANUAL contra o ano
#     (mesma base dos gráficos de barra). Reporta valor inicial (intercepto),
#     delta por ano (slope), R², dispersão residual e p -> tendência (↑/↓/≈).
#     Saída: tables/trend_metrics_over_time.txt (TSV) e .tex (longtable) +
#            tables/trend_summary_by_metric.{txt,tex}.
# =============================================================================
TABLE_DIR = 'paper/tables'
os.makedirs(TABLE_DIR, exist_ok=True)
P_SIG = 0.05


def _trend_row(sub, metric):
    s = sub.dropna(subset=[metric, 'SortieDateRef'])
    yearly = s.assign(Year=s['SortieDateRef'].dt.year).groupby('Year')[metric].mean()
    x = yearly.index.values.astype(float) - yearly.index.min()
    y = yearly.values.astype(float)
    n = len(y)
    if n < 3 or np.nanstd(y) < 1e-12:
        slope = 0.0
        intercept = float(y[0]) if n else np.nan
        r2, p, resid = 0.0, 1.0, 0.0
    else:
        lr = linregress(x, y)
        slope, intercept, r2, p = lr.slope, lr.intercept, lr.rvalue ** 2, lr.pvalue
        resid = float(np.sqrt(np.sum((y - (intercept + slope * x)) ** 2) / (n - 2)))
    trend = ('up' if (p < P_SIG and slope > 0) else
             'down' if (p < P_SIG and slope < 0) else 'flat')
    return {'N': n, 'Initial': intercept, 'Slope_per_year': slope,
            'R2': r2, 'Resid_SD': resid, 'p_value': p, 'Trend': trend}


_strat_order = _order_present(measures)
rows = []
for wl in sorted(measures.WL.unique()):
    for strat in _strat_order:
        sub = measures[(measures.WL == wl) & (measures.Strategy == strat)]
        if sub.empty:
            continue
        for metric in MEASURE_COLS:
            r = _trend_row(sub, metric)
            r.update({'WL': wl, 'Strategy': strat, 'Metric': metric})
            rows.append(r)

trend = pd.DataFrame(rows)
_mrank = {m: i for i, m in enumerate(MEASURE_COLS)}
_srank = {s: i for i, s in enumerate(_strat_order)}
trend = (trend.assign(_m=trend.Metric.map(_mrank), _s=trend.Strategy.map(_srank))
         .sort_values(['_m', '_s', 'WL']).drop(columns=['_m', '_s'])
         .reset_index(drop=True))
trend = trend[['WL', 'Strategy', 'Metric', 'N', 'Initial', 'Slope_per_year',
               'R2', 'Resid_SD', 'p_value', 'Trend']]

# ---- TSV (dados crus) ----
trend.to_csv(os.path.join(TABLE_DIR, f'trend_metrics_over_time_{EXP_PREFIX}.txt'),
             sep='\t', index=False, float_format='%.5f')

# ---- LaTeX longtable (matriz completa) ----
_ARROW = {'up': r'$\uparrow$', 'down': r'$\downarrow$', 'flat': r'$\approx$'}


def _disp_tex(s):
    return s.replace(' (Vol)', r'$_{\mathrm{Vol}}$')


tex = trend.copy()
tex['Strategy'] = tex['Strategy'].map(_disp_tex)
tex['Trend'] = tex['Trend'].map(_ARROW)
tex['WL'] = tex['WL'].map(lambda v: f'{v:.2f}')
for c in ['Initial', 'Slope_per_year', 'R2', 'Resid_SD']:
    tex[c] = tex[c].map(lambda v: f'{v:.4f}')
tex['p_value'] = tex['p_value'].map(lambda v: f'{v:.3f}')
tex = tex.rename(columns={'Slope_per_year': r'Slope/yr', 'R2': r'$R^2$',
                          'Resid_SD': r'Resid.\,SD', 'p_value': r'$p$'})
latex_full = tex.to_latex(
    index=False, longtable=True, escape=False, column_format='cllrrrrrrc',
    caption=('Linear modelling of each per-portfolio metric over time '
             '(yearly mean vs.\\ year), by window length and strategy. '
             r'Initial = fitted value at the first year; Slope/yr = change per year; '
             r'Trend: $\uparrow$/$\downarrow$ significant ($p<0.05$), $\approx$ otherwise.'),
    label='tab:trend_full')
with open(os.path.join(TABLE_DIR, f'trend_metrics_over_time_{EXP_PREFIX}.tex'), 'w') as f:
    f.write(latex_full)

# ---- Resumo por métrica (evidência de tendência) ----
summary = (trend.groupby('Metric')
           .agg(n_up=('Trend', lambda s: int((s == 'up').sum())),
                n_down=('Trend', lambda s: int((s == 'down').sum())),
                n_flat=('Trend', lambda s: int((s == 'flat').sum())),
                median_initial=('Initial', 'median'),
                median_slope_per_year=('Slope_per_year', 'median'),
                median_R2=('R2', 'median'))
           .reindex(MEASURE_COLS))
summary['n_contexts'] = trend.groupby('Metric').size().reindex(MEASURE_COLS)
summary = summary[['n_contexts', 'n_up', 'n_down', 'n_flat',
                   'median_initial', 'median_slope_per_year', 'median_R2']]
summary.to_csv(os.path.join(TABLE_DIR, f'trend_summary_by_metric_{EXP_PREFIX}.txt'),
               sep='\t', float_format='%.5f')

stex = summary.copy()
for c in ['median_initial', 'median_slope_per_year', 'median_R2']:
    stex[c] = stex[c].map(lambda v: f'{v:.4f}')
stex = stex.rename(columns={'n_contexts': '\\#ctx', 'n_up': r'$\uparrow$',
                            'n_down': r'$\downarrow$', 'n_flat': r'$\approx$',
                            'median_initial': 'Md.\\ Initial',
                            'median_slope_per_year': 'Md.\\ Slope/yr',
                            'median_R2': 'Md.\\ $R^2$'})
latex_summary = stex.to_latex(
    index=True, escape=False, column_format='lrrrrrrr',
    caption=('Per-metric trend evidence across all 60 (window length $\\times$ strategy) '
             'contexts: number of contexts with a significant positive/negative trend '
             r'($p<0.05$) or stable ($\approx$), and median initial value, slope/yr and $R^2$.'),
    label='tab:trend_summary')
with open(os.path.join(TABLE_DIR, f'trend_summary_by_metric_{EXP_PREFIX}.tex'), 'w') as f:
    f.write(latex_summary)

print(f'\nTabela de tendência: {len(trend)} linhas (WL x estratégia x métrica) '
      f'salvas em "{TABLE_DIR}/trend_metrics_over_time.txt" e ".tex".')
print('\n===== Resumo por métrica (contextos ↑ / ↓ / ≈ e medianas) =====')
print(summary.round(4).to_string())


#%%
# =============================================================================
# 4e) COMPARAÇÃO PAREADA ENTRE ESTRATÉGIAS (pareado por janela)
#     Para cada (WL, métrica), compara cada par de estratégias (pareado por
#     SortieDateRef) por DOIS testes:
#       - Wilcoxon signed-rank (não-paramétrico; direção "melhor" pela MEDIANA);
#       - t pareado de Student (ttest_rel; direção "melhor" pela MÉDIA).
#     Direção "melhor" por métrica:
#       maior é melhor -> Sharpe, Sortino, Mean Return;
#       menor é melhor -> Std, Pain Index, Sectoral Gini, Gini, Turnover;
#       Drawdown       -> menos negativo (menor perda) é melhor.
#     Saídas em tables/:
#       - pairwise_comparison.txt      : série pareada completa (TSV) com AMBOS os testes
#                                        (colunas sem prefixo = Wilcoxon; prefixo t_ = t pareado);
#       - wilcoxon_dominance.{txt,tex} : dominância (Wilcoxon);
#       - ttest_dominance.{txt,tex}    : dominância (t pareado).
# =============================================================================
HIGHER_BETTER = {'Sharpe Ratio', 'Sortino Ratio', 'Mean Return', 'Drawdown'}
LOWER_BETTER  = {'Standard Deviation', 'Pain Index', 'Sectoral Gini Index',
                 'Gini Index', 'Turnover Ratio'}
DIRECTION = {m: ('higher' if m in HIGHER_BETTER else 'lower') for m in MEASURE_COLS}
ALPHA = 0.05
MIN_PAIRS = 8

_wls = sorted(measures.WL.unique())
_pair_rows = []
# dominance[(metric, strat, wl)] = nº de estratégias significativamente superadas
_dom   = {(m, s, wl): 0 for m in MEASURE_COLS for s in _strat_order for wl in _wls}  # Wilcoxon
_dom_t = {(m, s, wl): 0 for m in MEASURE_COLS for s in _strat_order for wl in _wls}  # t pareado

for wl in _wls:
    sub = measures[measures.WL == wl]
    for metric in MEASURE_COLS:
        piv = sub.pivot_table(index='SortieDateRef', columns='Strategy', values=metric,
                              aggfunc='first')
        strats = [s for s in _strat_order if s in piv.columns]
        higher = metric in HIGHER_BETTER
        for a, b in combinations(strats, 2):
            pair = piv[[a, b]].dropna()
            x, y = pair[a].values, pair[b].values
            n = len(pair)
            ma, mb       = (np.median(x), np.median(y)) if n else (np.nan, np.nan)
            avg_a, avg_b = (np.mean(x),   np.mean(y))   if n else (np.nan, np.nan)
            if n < MIN_PAIRS or np.allclose(x, y):
                stat, p     = np.nan, 1.0
                t_stat, t_p = np.nan, 1.0
            else:
                try:
                    stat, p = wilcoxon(x, y)
                except ValueError:
                    stat, p = np.nan, 1.0
                try:
                    t_stat, t_p = ttest_rel(x, y)
                except ValueError:
                    t_stat, t_p = np.nan, 1.0
            better   = (a if ((ma > mb)       == higher) else b)   # Wilcoxon -> mediana
            t_better = (a if ((avg_a > avg_b) == higher) else b)   # t pareado -> média
            sig   = bool(p   < ALPHA) and n >= MIN_PAIRS
            t_sig = bool(t_p < ALPHA) and n >= MIN_PAIRS
            _pair_rows.append({'WL': wl, 'Metric': metric, 'A': a, 'B': b,
                               'Median_A': ma, 'Median_B': mb, 'Median_Diff': ma - mb,
                               'W': stat, 'p_value': p, 'Better': better,
                               'Significant': sig,
                               'Mean_A': avg_a, 'Mean_B': avg_b, 'Mean_Diff': avg_a - avg_b,
                               't_stat': t_stat, 't_p_value': t_p, 't_Better': t_better,
                               't_Significant': t_sig, 'N_pairs': n})
            if sig:
                _dom[(metric, better, wl)] += 1
            if t_sig:
                _dom_t[(metric, t_better, wl)] += 1

pairwise = pd.DataFrame(_pair_rows)
pairwise.to_csv(os.path.join(TABLE_DIR, f'pairwise_comparison_{EXP_PREFIX}.txt'),
                sep='\t', index=False, float_format='%.5f')


# ---- dominância: helpers genéricos (servem para Wilcoxon e para o t pareado) ----
def _dom_long(dom):
    rows = [{'Metric': m, 'Strategy': s, 'WL': wl, 'Direction': DIRECTION[m],
             'Wins_significant': dom[(m, s, wl)]}
            for (m, s, wl) in dom]
    df = pd.DataFrame(rows)
    return (df.assign(_m=df.Metric.map(_mrank), _s=df.Strategy.map(_srank))
            .sort_values(['_m', '_s', 'WL']).drop(columns=['_m', '_s']))


def _dom_matrix(metric, dom):
    mat = pd.DataFrame(0, index=_strat_order, columns=_wls, dtype=int)
    for s in _strat_order:
        for wl in _wls:
            mat.loc[s, wl] = dom[(metric, s, wl)]
    return mat


def _dom_to_latex(metric, dom, test_label, label_prefix):
    mat = _dom_matrix(metric, dom)
    disp = mat.astype(str)
    for col in mat.columns:                       # negrito no máximo da coluna (>0)
        mx = mat[col].max()
        if mx > 0:
            for idx in mat.index:
                if mat.loc[idx, col] == mx:
                    disp.loc[idx, col] = r'\textbf{' + str(mat.loc[idx, col]) + '}'
    disp.index = [_disp_tex(s) for s in disp.index]
    disp.columns = [f'{c:.2f}' for c in mat.columns]
    disp.index.name = 'Strategy'
    arrow = 'higher is better' if metric in HIGHER_BETTER else 'lower is better'
    note = (' (less negative = smaller loss)' if metric == 'Drawdown' else '')
    cap = (f'{test_label} dominance for {metric} '
           f'({arrow}{note}): number of strategies each one significantly '
           f'outperforms ($p<{ALPHA}$) per window length (WL, years). '
           f'Column-max in bold.')
    slug = metric.lower().replace(' ', '_')
    return disp.to_latex(escape=False, column_format='l' + 'c' * len(mat.columns),
                         caption=cap, label=f'tab:{label_prefix}_{slug}')


def _write_dominance(dom, name, test_label, label_prefix):
    _dom_long(dom).to_csv(os.path.join(TABLE_DIR, f'{name}_{EXP_PREFIX}.txt'),
                          sep='\t', index=False)
    latex = '\n\n'.join(_dom_to_latex(m, dom, test_label, label_prefix) for m in MEASURE_COLS)
    with open(os.path.join(TABLE_DIR, f'{name}_{EXP_PREFIX}.tex'), 'w') as f:
        f.write(latex)


_write_dominance(_dom,   'wilcoxon_dominance', 'Wilcoxon signed-rank', 'wilcox')
_write_dominance(_dom_t, 'ttest_dominance',    'Paired $t$-test',      'ttest')

# resumo no console: total de vitórias significativas por estratégia/métrica
_tot_w = (_dom_long(_dom).groupby(['Metric', 'Strategy'])['Wins_significant'].sum()
          .unstack().reindex(index=MEASURE_COLS, columns=_strat_order))
_tot_t = (_dom_long(_dom_t).groupby(['Metric', 'Strategy'])['Wins_significant'].sum()
          .unstack().reindex(index=MEASURE_COLS, columns=_strat_order))
print(f'\nComparação pareada: {len(pairwise)} pares salvos em '
      f'"{TABLE_DIR}/pairwise_comparison_{EXP_PREFIX}.txt" (Wilcoxon + t pareado).')
print('Dominância: "wilcoxon_dominance.{txt,tex}" (Wilcoxon) e "ttest_dominance.{txt,tex}" (t).')
print('\n===== Vitórias significativas (soma sobre WL) — Wilcoxon =====')
print(_tot_w.to_string())
print('\n===== Vitórias significativas (soma sobre WL) — t pareado =====')
print(_tot_t.to_string())


#%%
# =============================================================================
# 4f) TABELA GERAL: estratégia COMPOSTA (série temporal) vs MÉDIA DAS CARTEIRAS
#     Por estratégia e WL. Métricas baseadas em retorno computadas na série
#     diária composta (com e sem custo) usando as MESMAS definições de Measures.py;
#     médias das carteiras individuais com IC 95% (t de Student). Métricas de peso
#     (Gini, Sectoral Gini, Turnover) só existem por carteira.
#     Notação científica padronizada (2 casas na mantissa).
#     Saídas: tables/summary_master.txt (TSV) e summary_master.tex (longtable).
# =============================================================================
_COMPOSED_FUNCS = {
    'Sharpe Ratio':       lambda r, rf: Measures.sharpe_ratio(series=r, risk_free_rate=rf),
    'Sortino Ratio':      lambda r, rf: Measures.sortino_ratio(series=r, risk_free_rate=rf),
    'Standard Deviation': lambda r, rf: Measures.standard_deviation(series=r),
    'Mean Return':        lambda r, rf: Measures.mean_return(series=r),
    'Drawdown':           lambda r, rf: Measures.drawdown(series=r),
    'Pain Index':         lambda r, rf: Measures.pain_index(series=r),
}


def _composed(series, rf, fn):
    try:
        return float(fn(series.dropna(), rf))
    except Exception:
        return np.nan


master_rows = []
for wl in _wls:
    res = results.get(wl)
    sub_meas = measures[measures.WL == wl]
    for strat in _strat_order:
        comp_nc, comp_c = {}, {}
        if res is not None and strat in res['returns'].columns:
            r_nc = res['returns'][strat]
            rf_nc = res['returns']['SELIC'] if 'SELIC' in res['returns'].columns else 0.0
            r_c = res['returns_cost'][strat] if strat in res['returns_cost'].columns else None
            rf_c = res['returns_cost']['SELIC'] if 'SELIC' in res['returns_cost'].columns else rf_nc
            for m, fn in _COMPOSED_FUNCS.items():
                comp_nc[m] = _composed(r_nc, rf_nc, fn)
                comp_c[m] = _composed(r_c, rf_c, fn) if r_c is not None else np.nan

        g = sub_meas[sub_meas.Strategy == strat]
        for metric in MEASURE_COLS:
            vals = g[metric].dropna()
            n = len(vals)
            if n > 1:
                mean = vals.mean()
                hw = _student_t.ppf(0.975, n - 1) * vals.std(ddof=1) / np.sqrt(n)
                lo, hi = mean - hw, mean + hw
            elif n == 1:
                mean = lo = hi = float(vals.iloc[0]); hw = 0.0
            else:
                mean = lo = hi = hw = np.nan
            master_rows.append({
                'WL': wl, 'Strategy': strat, 'Metric': metric,
                'Composed_NoCost': comp_nc.get(metric, np.nan),
                'Composed_Cost': comp_c.get(metric, np.nan),
                'Portfolio_Mean': mean, 'Portfolio_CI95_HW': hw,
                'Portfolio_CI95_low': lo, 'Portfolio_CI95_high': hi, 'Portfolio_n': n})

master = pd.DataFrame(master_rows)
master = (master.assign(_m=master.Metric.map(_mrank), _s=master.Strategy.map(_srank))
          .sort_values(['_m', '_s', 'WL']).drop(columns=['_m', '_s']).reset_index(drop=True))
master = master[['WL', 'Strategy', 'Metric', 'Composed_NoCost', 'Composed_Cost',
                 'Portfolio_Mean', 'Portfolio_CI95_HW', 'Portfolio_CI95_low',
                 'Portfolio_CI95_high', 'Portfolio_n']]
master.to_csv(os.path.join(TABLE_DIR, f'summary_master_{EXP_PREFIX}.txt'),
              sep='\t', index=False, float_format='%.4e')


# ---- LaTeX (notação científica padronizada, 2 casas) ----
def _sci(v, dec=2):
    if pd.isna(v):
        return '--'
    if v == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(v))))
    mant = v / 10.0 ** exp
    return f'${mant:.{dec}f}\\times10^{{{exp}}}$'


mtex = pd.DataFrame({
    'WL': master.WL.map(lambda v: f'{v:.2f}'),
    'Strategy': master.Strategy.map(_disp_tex),
    'Metric': master.Metric,
    'Comp.\\ (no cost)': master.Composed_NoCost.map(_sci),
    'Comp.\\ (cost)': master.Composed_Cost.map(_sci),
    'PF mean': master.Portfolio_Mean.map(_sci),
    r'PF $\pm$95\% CI': master.Portfolio_CI95_HW.map(_sci),
})
latex_master = mtex.to_latex(
    index=False, longtable=True, escape=False, column_format='cllrrrr',
    caption=('General summary by strategy and window length (WL, years). '
             'Composed = metric on the full daily return series of the composed strategy '
             '(no cost / net of cost), using the Measures definitions; PF mean = mean across '
             'individual rebalanced portfolios with 95\\% confidence interval (Student-$t$ '
             'half-width). Weight-based metrics (Gini, Sectoral Gini, Turnover) exist only '
             'at portfolio level (composed = ``--\'\'). Values in scientific notation.'),
    label='tab:summary_master')
with open(os.path.join(TABLE_DIR, f'summary_master_{EXP_PREFIX}.tex'), 'w') as f:
    f.write(latex_master)

print(f'\nTabela geral: {len(master)} linhas (WL x estratégia x métrica) salvas em '
      f'"{TABLE_DIR}/summary_master.txt" e ".tex".')
print('\n===== Amostra (WL=1.00, métricas-chave) — composta vs média das carteiras =====')
_demo = master[(master.WL == 1.00) &
               (master.Metric.isin(['Sharpe Ratio', 'Standard Deviation', 'Mean Return']))]
print(_demo[['Strategy', 'Metric', 'Composed_NoCost', 'Composed_Cost',
             'Portfolio_Mean', 'Portfolio_CI95_HW']].round(4).to_string(index=False))


#%%
# =============================================================================
# 4g) CORRELAÇÕES ENTRE SÉRIES DE RETORNO DIÁRIO
#     (A) entre estratégias DIFERENTES, para cada WL  -> 1 matriz por painel,
#         grade 5x2 (A)..(J), uma por window length;
#     (B) da MESMA estratégia entre WL DIFERENTES      -> 1 matriz por painel,
#         grade 2x3 (A)..(F), uma por estratégia.
#     Base: retornos diários BRUTOS (sem custo). O ajuste de custo é um pico
#     esparso aplicado só ao 1o dia de cada holding (89 dias) e distorceria a
#     co-variação; a série bruta isola a co-movimentação intrínseca. Correlação
#     de Pearson (pandas .corr(), pares completos). SELIC excluída (benchmark).
#     Saídas: figures/correlacoes/corr_strategies_by_wl.{pdf,png}
#             figures/correlacoes/corr_wl_by_strategy.{pdf,png}
# =============================================================================
CORR_DIR = 'paper/figures/correlacoes'
os.makedirs(CORR_DIR, exist_ok=True)
CORR_KEY = 'returns'        # 'returns' (bruto) | 'returns_cost' (líquido de custo)
CORR_CMAP = 'viridis'


def _fmt_corr(v):
    """0.95 -> '.95'; 1.00 -> '1.00' (compacta o zero à esquerda)."""
    t = f'{v:.2f}'
    if t.startswith('0.'):
        return t[1:]
    if t.startswith('-0.'):
        return '-' + t[2:]
    return t


def _heatmap(ax, M, xlabels, ylabels, vmin, vmax, fs=6.5):
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=CORR_CMAP, aspect='equal')
    ax.set_xticks(range(len(xlabels)))
    ax.set_yticks(range(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=fs)
    ax.set_yticklabels(ylabels, fontsize=fs)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    span = (vmax - vmin) or 1.0
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            v = M[r, c]
            if np.isnan(v):
                continue
            nz = (min(max(v, vmin), vmax) - vmin) / span
            ax.text(c, r, _fmt_corr(v), ha='center', va='center',
                    fontsize=fs - 0.5, color=('white' if nz < 0.55 else 'black'))
    return im


def _offdiag(C):
    m = C.values
    return m[~np.eye(m.shape[0], dtype=bool)]


def _shared_range(off_list):
    allv = np.concatenate(off_list)
    vmin = np.floor(np.nanmin(allv) * 20) / 20    # arredonda p/ baixo a 0.05
    vmax = np.ceil(np.nanmax(allv) * 20) / 20     # diagonal (=1) satura no topo
    if vmax - vmin < 0.1:
        vmin = max(0.0, vmin - 0.05)
    return vmin, vmax


_RET_LABEL = 'gross of costs' if CORR_KEY == 'returns' else 'net of costs'

# ----- (A) correlação entre estratégias, por WL -----------------------------
# usa só os WL da grade (ExperimentoOHRP -> 1,2,3,4,5 => 1x5)
_wls_corr = sorted(w for w in results if w in GRID_WLS)
corrA, offA = {}, []
for wl in _wls_corr:
    df = results[wl][CORR_KEY]
    cols = [c for c in STRAT_ORDER if c in df.columns]
    C = df[cols].corr()
    corrA[wl] = C
    offA.append(_offdiag(C))
vminA, vmaxA = _shared_range(offA)

nA = len(_wls_corr)
ncolsA = 5
nrowsA = int(np.ceil(nA / ncolsA))
figA, axesA = plt.subplots(nrowsA, ncolsA, figsize=(2.7 * ncolsA + 0.6, 3.0 * nrowsA + 0.9),
                           layout='constrained')
axesA = np.atleast_1d(axesA).ravel()
imA = None
for i, (wl, ax) in enumerate(zip(_wls_corr, axesA)):
    C = corrA[wl]
    labels = [disp(c) for c in C.columns]
    imA = _heatmap(ax, C.values, labels, labels, vminA, vmaxA, fs=5.5)
    ax.set_title(f'({_LETTERS[i]})  WL = {wl:.2f} yr', fontsize=8.5,
                 fontweight='bold')
for ax in axesA[nA:]:
    ax.set_visible(False)

figA.suptitle('Correlation between strategies (daily returns, '
              f'{_RET_LABEL}), by window length',
              fontsize=10, fontweight='bold')
cbA = figA.colorbar(imA, ax=axesA.tolist(), shrink=0.55, aspect=30, pad=0.02)
cbA.set_label('Pearson correlation', fontsize=8.5)
cbA.ax.tick_params(labelsize=7.5)
_baseA = os.path.join(CORR_DIR, f'corr_strategies_by_wl_{EXP_PREFIX}')
figA.savefig(_baseA + '.pdf'); figA.savefig(_baseA + '.png')
plt.close(figA)

# ----- (B) correlação da mesma estratégia entre WL --------------------------
_strats_corr = [s for s in STRAT_ORDER
                if any(s in results[wl][CORR_KEY].columns for wl in _wls_corr)]
corrB, offB = {}, []
for s in _strats_corr:
    cols = {f'{wl:.2f}': results[wl][CORR_KEY][s]
            for wl in _wls_corr if s in results[wl][CORR_KEY].columns}
    frame = pd.DataFrame(cols)
    C = frame.corr()
    corrB[s] = C
    offB.append(_offdiag(C))
vminB, vmaxB = _shared_range(offB)

nB = len(_strats_corr)
ncolsB = 2 if nB <= 4 else 3          # 4 estratégias (ExperimentoOHRP) -> 2x2, sem painéis vazios
nrowsB = int(np.ceil(nB / ncolsB))
figB, axesB = plt.subplots(nrowsB, ncolsB, figsize=(2.8 * ncolsB, 2.7 * nrowsB + 0.9),
                           layout='constrained')
axesB = np.atleast_1d(axesB).ravel()
imB = None
for i, (s, ax) in enumerate(zip(_strats_corr, axesB)):
    C = corrB[s]
    labels = list(C.columns)
    imB = _heatmap(ax, C.values, labels, labels, vminB, vmaxB, fs=5.0)
    ax.set_title(f'({_LETTERS[i]})  {disp(s)}', fontsize=8.5, fontweight='bold')
for ax in axesB[nB:]:
    ax.set_visible(False)

figB.suptitle('Correlation of each strategy across window lengths '
              f'(daily returns, {_RET_LABEL})',
              fontsize=10, fontweight='bold')
figB.supxlabel('Window length (years)', fontsize=8.5)
cbB = figB.colorbar(imB, ax=axesB.tolist(), shrink=0.6, aspect=30, pad=0.02)
cbB.set_label('Pearson correlation', fontsize=8.5)
cbB.ax.tick_params(labelsize=7.5)
_baseB = os.path.join(CORR_DIR, f'corr_wl_by_strategy_{EXP_PREFIX}')
figB.savefig(_baseB + '.pdf'); figB.savefig(_baseB + '.png')
plt.close(figB)

print(f'\nFiguras de correlação salvas em "{CORR_DIR}/" (sufixo _{EXP_PREFIX}, PDF + PNG):')
print(f'  - corr_strategies_by_wl  : {nrowsA}x{ncolsA} ({nA} WL) x matriz '
      f'{len(corrA[_wls_corr[0]])}x{len(corrA[_wls_corr[0]])} entre estratégias '
      f'(range off-diag {vminA:.2f}–{vmaxA:.2f}).')
print(f'  - corr_wl_by_strategy    : {nB} estratégias ({nrowsB}x{ncolsB}) x matriz '
      f'{nA}x{nA} entre WL (range off-diag {vminB:.2f}–{vmaxB:.2f}).')


#%%
# =============================================================================
# 5) COMENTÁRIOS SOBRE AS MEDIDAS POR CARTEIRA
# =============================================================================
# (S-OHRP = antigo SOHRP;  S-OHRP_Vol = variante com vol-target)
#
# TURNOVER — o ponto que mais chama atenção
#   • S-OHRP é, de longe, a de MAIOR giro: ~0.55 de turnover médio por carteira,
#     contra ~0.18 do HRP, ~0.40 do OHRP e só ~0.07–0.10 de EW e RP. O "S-"
#     (etapa de otimização sobre o HRP) ~triplica o giro do HRP puro. É isso que
#     explica por que o S-OHRP é o mais penalizado na série COM custo: as medidas
#     por carteira antecipam o gap bruto-vs-líquido visto nas séries temporais.
#   • S-OHRP_Vol reduz o turnover de ~0.55 para ~0.38 mantendo o Sharpe — o
#     vol-target é também redução de custo, não só de risco.
#
# RISCO (Std Dev, Drawdown, Pain Index)
#   • OHRP é a mais defensiva (menor desvio ~0.144, menor drawdown ~-0.055);
#     HRP logo atrás. EW é a pior (desvio ~0.20, drawdown ~-0.079).
#   • S-OHRP_Vol corta o desvio do S-OHRP de ~0.185 p/ ~0.16 e o drawdown de
#     ~-0.071 p/ ~-0.062.
#
# RETORNO E SHARPE POR CARTEIRA
#   • Retorno médio por carteira é parecido (~0.13–0.14); a diferenciação vem do
#     risco. Sharpe médio por carteira: HRP (~0.57) ≈ OHRP ≈ S-OHRP (~0.55) >
#     RP (~0.47) > EW (~0.40).
#   • Consistência: HRP e S-OHRP têm a maior fração de carteiras com Sharpe
#     positivo (~57–58%) vs EW (~54%). EW tem mediana (~0.17) << média (~0.40)
#     → puxada por poucos períodos; S-OHRP/HRP têm média≈mediana (mais regulares).
#   • O Sharpe por carteira é MUITO ruidoso (desvio ~2.7; -6 a +11) pois cada
#     carteira cobre ~2 meses — ler como tendência média. É coerente com o Sharpe
#     das séries completas (família HRP > RP > EW).
#
# CONCENTRAÇÃO (Gini de pesos e Gini setorial)
#   • Gini de pesos: EW = 0 (pesos iguais). OHRP é a mais concentrada (~0.71),
#     S-OHRP (~0.60); HRP (~0.39) e RP (~0.24) mais distribuídas.
#   • Gini SETORIAL: OHRP e S-OHRP concentram mais por setor (~0.45–0.47); RP é a
#     mais diversificada (~0.31). As de melhor risco/Sharpe são as mais
#     concentradas — relevante para limites de risco do paper.
#
# DEPENDÊNCIA DO WINDOW LENGTH
#   • S-OHRP: Sharpe por carteira tem pico em WL≈1.0 (~0.77) e decai p/ ~0.36 em
#     WL=2.0; turnover cai com WL (janela maior → estimativa estável → menos
#     rebalanceamento). WL≈1.0 é o "sweet spot".
#   • S-OHRP_Vol é bem mais ESTÁVEL ao longo do WL (desvio ~0.16 em todos),
#     reforçando que o vol-target reduz a sensibilidade ao hiperparâmetro.
#
# SÍNTESE / aderência às séries temporais
#   • As medidas por carteira CONDIZEM com as séries: explicam o brilho do S-OHRP
#     no bruto e a queda no líquido (turnover), a eficiência de risco de OHRP/HRP
#     e o atraso do EW. O S-OHRP_Vol surge como "best-of-both" (risco e custo
#     menores, Sharpe equivalente) — bom candidato a estratégia headline.

print('\nVer comentários (seção 5) no final do arquivo.')
# %%

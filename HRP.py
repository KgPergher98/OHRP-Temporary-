"""
HRP - Hierarchical Risk Parity
==============================

Implementation of the allocation method of de Prado (2016), used both as a
benchmark and as the allocation engine that OHRP and S-OHRP feed with projected
returns.

Three stages
------------
    1. Hierarchical tree clustering of the correlation-derived distance matrix
    2. Quasi-diagonalization via optimal leaf ordering
    3. Recursive bisection, splitting weights between sibling clusters in
       inverse proportion to their variance

Optional `min_weight` / `max_weight` bounds are enforced afterwards by an
iterative clip-and-redistribute loop.

Usage
-----
    from HRP import HRP

    model = HRP(Y=returns, correlation_method="pearson", linkage_method="ward")
    model.run()
    model.weights
"""

import numpy
import pandas
import scipy

class HRP():

    def Risk(w = None, cov = None):
        w_ = numpy.array(w, ndmin = 2)
        cov_ = numpy.array(cov, ndmin=2) # MATRIZ DE COVARIANCIA
        risk = w_.T @ cov_ @ w_          
        risk = numpy.sqrt(risk.item())   # DESVIO PADRÃO
        return risk

    def naive_risk(returns, cov):

        assets = returns.columns.tolist() # ATIVOS ANALISADOS
        n = len(assets) # QTD DE ATIVOS
        inv_risk = numpy.zeros((n, 1)) # INICIALIZA VETOR COLUNA COM ZEROS

        for i in assets:
            k = assets.index(i)
            w = numpy.zeros((n, 1))
            w[k, 0] = 1
            # CONSIDERA APENAS A CONTRIBUICAO DO ATIVO INDIVIDUAL
            w = pandas.DataFrame(w, columns = ["weights"], index = assets)
            risk = HRP.Risk(w = w, cov = cov)
            inv_risk[k, 0] = risk

        # PESO DEFAULT É O INVERSO DA VARIÂNCIA DO ATIVO
        inv_risk = 1 / numpy.power(inv_risk, 2)
        # NORMALIZA - SOMA DO VETOR = 1
        weights = inv_risk * (1 / numpy.sum(inv_risk))
        weights = weights.reshape(-1, 1)
        return weights

    def recursive_bisection(self, asset_list, sort_order):
        # INICIA UM VETOR W0 COM TODOS OS TERMOS Wi = 1.0
        weights = pandas.Series(1.0, index = asset_list)
        # INICIALIZA ORDEM DAS FOLHAS JÁ CALCULADA
        items = [sort_order]
        # ITERA ATÉ ATINGIR O FATOR DE ALOCAÇÃO NA ULTIMA FOLHA (LEAF)
        while len(items) > 0:
            # VAI REPARTINDO O CONJUNTO EM DOIS SEMPRE (DIREITA E ESQUERDA)
            # INDIFERENTEMENTE DA ESTRUTURA, OS PARES JÁ SÃO MAXIMIZADOS EM DISSIMILARIDADE
            # PROCESSO SEGUE ATÉ QUE NÃO HAJA MAIS GRUPO PARA REPARTIR
            items = [i[j:k] for i in items
                for j, k in (
                    (0, len(i) // 2),
                    (len(i) // 2, len(i)),
                ) if len(i) > 1
            ] # LISTA DE ARRAYS
            # ALOCAÇÃO DE PESOS ENTRE CLUSTERS DA DIREITA E ESQUERDA
            for i in range(0, len(items), 2):
                left_cluster = items[i]      # CLUSTER DA ESQUERDA, LISTA DE INDICES
                right_cluster = items[i + 1] # CLUSTER DA DIREITA , LISTA DE INDICES
            
                # ESQUERDA
                left_cov = self.covar.iloc[left_cluster, left_cluster]         # COVARIANCIA
                left_returns = self.Y.iloc[:, left_cluster]                    # RETORNOS  
                left_weights = HRP.naive_risk(left_returns, left_cov)          # PESOS INICIAIS - MIN VAR

                left_risk = HRP.Risk(
                    w = left_weights,
                    cov = left_cov
                )
                left_risk = numpy.power(left_risk, 2)

                # DIREITA
                right_cov = self.covar.iloc[right_cluster, right_cluster]
                right_returns = self.Y.iloc[:, right_cluster]
                right_weights = HRP.naive_risk(right_returns, right_cov)

                right_risk = HRP.Risk(
                    w = right_weights,
                    cov = right_cov
                )
                right_risk = numpy.power(right_risk, 2)

                # ALOCAÇÃO DE PESOS - PROPORCIONAL AO RISCO
                alpha_1 = 1 - left_risk / (left_risk + right_risk)

                weights.iloc[left_cluster]  *= alpha_1       # PESOS DA ESQUERDA
                weights.iloc[right_cluster] *= 1 - alpha_1   # PESOS DA DIREITA

        weights = pandas.DataFrame(weights)
        weights.columns = ["weights"]
        return weights

    def __init__(self, Y, linkage_method:str = 'single', correlation_method:str = 'pearson', min_weight:float = 0, max_weight:float = 1.0):
        # Y - RETORNOS A SEREM UTILIZADOS
        self.Y = Y

        if self.Y.shape[0] <= 1:
            raise ValueError("The returns DataFrame Y must have more than one row (time periods).")

        # COVARIANCIA DOS RETORNOS
        self.covar = pandas.DataFrame(
            numpy.cov(self.Y, rowvar = False), 
            index = self.Y.columns, 
            columns = self.Y.columns
        )
        # MÉDIA TEMPORAL DOS RETORNOS
        self.means = pandas.DataFrame(
            numpy.mean(self.Y, axis = 0)
        ).transpose()

        # CORRELÇÃO DE PEARSON (CODEPENDENCIA)
        self.codep = self.Y.corr(method = correlation_method)

        # MATRIZ DE DISTANCIA
        # TRANSFORMAMOS A CORRELAÇÃO EM UMA MÉTRICA GEOMÉTRICA/MEDIDA DE DISTANCIA
        self.distance = numpy.sqrt(numpy.clip((1 - self.codep)/2, a_min = 0, a_max = 1.0))
        self.distance = scipy.spatial.distance.squareform(self.distance, checks = False)

        # CLUSTERIZAÇÃO HIERARQUICA
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
        # MATRIZ DE LINKAGE Z: DIM (n - 1) X 4
        # Z[:, 0] - INDICE 1o CLUSTER/ATIVO A SER FUNDIDO
        # Z[:, 1] - INDICE 2o CLUSTER/ATIVO A SER FUNDIDO
        # Z[:, 2] - DISTANCIA/DISSIMILARIDADE ENTRE OS ATIVOS
        # Z[:, 3] - NUMERO DE DADOS NO CLUSTER FORMADO
        self.clustering = scipy.cluster.hierarchy.linkage(
            self.distance, 
            method = linkage_method, # METODO DE LINKAGE
            optimal_ordering = True  # ORDENAÇÃO DAS FOLHAS - QUASI DIAGONALIZAÇÃO
        )
        # [1] Daniel Mullner, "Modern hierarchical, agglomerative clustering algorithms", :arXiv:`1109.2378v1`.
        # [2] Ziv Bar-Joseph, David K. Gifford, Tommi S. Jaakkola, "Fast optimal leaf ordering for hierarchical clustering", 2001. Bioinformatics :doi:`10.1093/bioinformatics/17.suppl_1.S22`
        self.min_weight = min_weight
        self.max_weight = max_weight
        
    def run(self):
        assets_list = self.Y.columns.tolist()
        # FOLHAS ORDENADAS, CONFORME JUNÇÕES NA MATRIZ Z
        ordering_assets_list = scipy.cluster.hierarchy.leaves_list(self.clustering)
        # CONVERTE O INDICE DA FOLHA NOS "ATIVOS" INPUTADOS
        ordered_assets = [assets_list[i] for i in ordering_assets_list]
        # REINDEXAÇÃO DA MATRIZ DE CODEPENDENCIA
        self.codep = self.codep.reindex(index = ordered_assets, columns = ordered_assets)
        # BISSECÇÃO RECURSIVA
        self.weights = HRP.recursive_bisection(
            self, 
            asset_list = assets_list, 
            sort_order = ordering_assets_list
        ).weights

        # CONTROLE DOS PESOS
        if (self.max_weight < 1) or (self.min_weight > 0):

            upper_bound = (self.weights.copy() * 0) + self.max_weight
            lower_bound = (self.weights.copy() * 0) + self.min_weight
            if (upper_bound < self.weights).any().item() or (lower_bound > self.weights).any().item():
                max_iter = 100
                j = 0
                while ((upper_bound < self.weights).any().item() or (lower_bound > self.weights).any().item()) and (j < max_iter):
                    
                    weights_original = self.weights.copy()
                    self.weights = numpy.maximum(numpy.minimum(self.weights, upper_bound), lower_bound)
                    tickers_mod = self.weights[(self.weights < upper_bound) & (self.weights > lower_bound)].index.tolist()

                    weights_add = numpy.maximum(weights_original - upper_bound, 0).sum()
                    weights_sub = numpy.minimum(weights_original - lower_bound, 0).sum()
                    delta = weights_add + weights_sub

                    if delta != 0:
                        self.weights[tickers_mod] += (delta * self.weights[tickers_mod] / self.weights[tickers_mod].sum())

                    j += 1

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Limpa o terminal de forma compatível com Windows ('cls') e Linux/Mac ('clear')
os.system('cls' if os.name == 'nt' else 'clear')

# ==============================================================================
# 1. OBTENÇÃO E TRATAMENTO DOS DADOS
# ==============================================================================
try:
    print('Obtendo os dados do ISP-RJ...')

    # Endereço oficial da base de evolução mensal por CISP (Delegacia)
    ENDERECO_DADOS = 'https://www.ispdados.rj.gov.br/Arquivos/BaseDPEvolucaoMensalCisp.csv'

    # Leitura do arquivo CSV com separador ';' e codificação 'iso-8859-1'
    df_ocorrencias = pd.read_csv(ENDERECO_DADOS, sep=';', encoding='iso-8859-1')

    # Seleciona as colunas de interesse: Delegacia (cisp), Município (munic) e Recuperação de Veículos
    df_rec = df_ocorrencias[['cisp', 'munic', 'recuperacao_veiculos']]

    # Agrupa por delegacia (cisp) e município, somando o total histórico de recuperações
    df_rec = df_rec.groupby(['cisp', 'munic'], as_index=False)['recuperacao_veiculos'].sum()

    # Ordena os dados pelo número da delegacia (cisp)
    df_rec = df_rec.sort_values(by='cisp')

    # Calcula o total geral de veículos recuperados no estado
    total_rec = df_rec['recuperacao_veiculos'].sum()

    # Calcula o percentual de participação de cada delegacia sobre o total
    df_rec['percentual %'] = round(
        ((df_rec['recuperacao_veiculos'] / total_rec) * 100), 2
    )

    print(df_rec)

except Exception as e:
    print(f'Erro ao obter os dados: {e}')

# ==============================================================================
# 2. MEDIDAS DE TENDÊNCIA CENTRAL (MÉDIA E MEDIANA)
# ==============================================================================
try:
    # Converte a coluna de recuperações para um array NumPy para cálculos estatísticos
    array_rec = np.array(df_rec['recuperacao_veiculos'])

    media_rec = np.mean(array_rec)
    mediana_rec = np.median(array_rec)

    # Calcula a distância percentual entre a média e a mediana para avaliar distorção
    distancia = abs((media_rec - mediana_rec) / mediana_rec * 100)

    print('\nMedidas de Tendência Central')
    print(30 * '-')
    print(f'Média: {media_rec:.2f}')
    print(f'Mediana: {mediana_rec:.2f}')
    print(f'Distância entre média e mediana: {distancia:.2f}%')

except Exception as e:
    print(f'Erro ao calcular tendência central: {e}')

# ==============================================================================
# 3. MEDIDAS DE POSIÇÃO (QUARTIS) E SEGMENTAÇÃO (MENORES E MAIORES)
# ==============================================================================
try:
    # Calcula o 1º Quartil (25%), 2º Quartil/Mediana (50%) e 3º Quartil (75%)
    q1 = np.quantile(array_rec, 0.25)
    q2 = np.quantile(array_rec, 0.50)
    q3 = np.quantile(array_rec, 0.75)

    print('\nMedidas de Posição')
    print(30 * '-')
    print(f'Q1 (25%): {q1}')
    print(f'Q2 (50%): {q2}')
    print(f'Q3 (75%): {q3}')

    # Filtra delegacias abaixo de Q1 (25% com menores recuperações)
    df_rec_menores = df_rec[df_rec['recuperacao_veiculos'] < q1]

    # Filtra delegacias acima de Q3 (25% com maiores recuperações)
    df_rec_maiores = df_rec[df_rec['recuperacao_veiculos'] > q3]

    # Exibe e exporta as delegacias com menores recuperações
    print('\nDelegacias com MENORES recuperações de veículos (< Q1)')
    print(30 * '-')
    print(df_rec_menores.sort_values(by='recuperacao_veiculos', ascending=True))
    df_rec_menores.to_csv('delegacias_menores_recuperacoes.csv', index=False, sep=';', encoding='iso-8859-1')

    # Exibe e exporta as delegacias com maiores recuperações
    print('\nDelegacias com MAIORES recuperações de veículos (> Q3)')
    print(30 * '-')
    print(df_rec_maiores.sort_values(by='recuperacao_veiculos', ascending=False))
    df_rec_maiores.to_csv('delegacias_maiores_recuperacoes.csv', index=False, sep=';', encoding='iso-8859-1')

except Exception as e:
    print(f'Erro ao analisar quartis e distribuição: {e}')

# ==============================================================================
# 4. MEDIDAS DE DISPERSÃO, ASSIMETRIA E CURTOSE
# ==============================================================================
try:
    maximo = np.max(array_rec)
    minimo = np.min(array_rec)
    amplitude = maximo - minimo

    variancia = np.var(array_rec)
    distancia_var_media = (variancia / (media_rec ** 2)) * 100
    desvio_padrao = np.std(array_rec)
    coef_variacao = (desvio_padrao / media_rec) * 100

    # Assimetria (Skewness) e Curtose (Kurtosis)
    assimetria = df_rec['recuperacao_veiculos'].skew()
    curtose = df_rec['recuperacao_veiculos'].kurtosis()

    print('\nAnalisando a Distribuição (Assimetria e Curtose)')
    print(30 * '-')
    print(f'Assimetria: {assimetria:.2f}')
    print(f'Curtose: {curtose:.2f}')

except Exception as e:
    print(f'Erro ao calcular dispersão, assimetria e curtose: {e}')

# ==============================================================================
# 5. Identificando os Outliers (Valor Discrepante)
# ==============================================================================
try:
    # Intervalo Interquartil (IQR = Q3 - Q1)
    iqr = q3 - q1

    # Cálculo dos limites inferior e superior para detecção de outliers
    limite_superior = q3 + 1.5 * iqr
    limite_inferior = q1 - 1.5 * iqr

    print('\nLimites para Detecção de Outliers')
    print(30 * '-')
    print(f'IQR: {iqr}')
    print(f'Limite Inferior: {limite_inferior:.2f}')
    print(f'Limite Superior: {limite_superior:.2f}')

    # Outliers inferiores (muito abaixo do padrão)
    df_rec_outliers_menores = df_rec[df_rec['recuperacao_veiculos'] < limite_inferior]
    print('\nDelegacias com Outliers Inferiores')
    print(30 * '-')
    if len(df_rec_outliers_menores) == 0:
        print('Não há outliers inferiores (pois o limite inferior é negativo).')
    else:
        print(df_rec_outliers_menores)

    # Outliers superiores (muito acima do padrão - discrepantes)
    df_rec_outliers_maiores = df_rec[df_rec['recuperacao_veiculos'] > limite_superior].sort_values(
        by='recuperacao_veiculos', ascending=False
    )
    print('\nDelegacias com Outliers Superiores (Discrepantes)')
    print(30 * '-')
    if len(df_rec_outliers_maiores) == 0:
        print('Não há outliers superiores.')
    else:
        print(df_rec_outliers_maiores)

except Exception as e:
    print(f'Erro ao calcular os outliers: {e}')

# ==============================================================================
# 6. RESUMO CONSOLIDADO DAS MEDIDAS ESTATÍSTICAS
# ==============================================================================
try:
    print('\nResumo Geral das Medidas Estatísticas')
    print(30 * '-')
    print(f'Mínimo: {minimo}')
    print(f'Limite Inferior: {limite_inferior:.2f}')
    print(f'Q1: {q1:.2f}')
    print(f'Q2 (Mediana): {q2:.2f}')
    print(f'Q3: {q3:.2f}')
    print(f'Limite Superior: {limite_superior:.2f}')
    print(f'Máximo: {maximo}')
    print(f'Total de Veículos Recuperados: {total_rec}')
    print(f'Média: {media_rec:.2f}')
    print(f'Distância Média x Mediana: {distancia:.2f}%')
    print(f'IQR: {iqr:.2f}')
    print(f'Amplitude Total: {amplitude}')
    print(f'Variância: {variancia:.2f}')
    print(f'Distância Variância e Média: {distancia_var_media:.2f}%')
    print(f'Desvio Padrão: {desvio_padrao:.2f}')
    print(f'Coeficiente de Variação: {coef_variacao:.2f}%')

except Exception as e:
    print(f'Erro ao exibir resumo das medidas: {e}')

# ==============================================================================
# 7. PAINEL VISUAL (BOXPLOT, MÉTRICAS, HISTOGRAMA E TOP 10 DELEGACIAS)
# ==============================================================================
try:
    plt.figure(figsize=(14, 10))

    # Subplot 1: Boxplot horizontal para evidenciar quartis e outliers
    plt.subplot(2, 2, 1)
    plt.boxplot(df_rec['recuperacao_veiculos'], orientation='horizontal', showmeans=True)
    plt.title('Distribuição das Recuperações (Boxplot)')
    plt.xlabel('Quantidade de Veículos Recuperados')
    plt.ylabel('Delegacias (CISP)')

    # Subplot 2: Quadro resumo das medidas estatísticas
    plt.subplot(2, 2, 2)
    plt.text(0.05, 0.92, f'Limite Inferior: {limite_inferior:.2f}', fontsize=11)
    plt.text(0.05, 0.85, f'Mínimo: {minimo}', fontsize=11)
    plt.text(0.05, 0.78, f'Q1 (25%): {q1:.2f}', fontsize=11)
    plt.text(0.05, 0.71, f'Q2 (Mediana): {q2:.2f}', fontsize=11)
    plt.text(0.05, 0.64, f'Q3 (75%): {q3:.2f}', fontsize=11)
    plt.text(0.05, 0.57, f'Limite Superior: {limite_superior:.2f}', fontsize=11)
    plt.text(0.05, 0.50, f'Máximo: {maximo}', fontsize=11)
    plt.text(0.05, 0.40, f'Média: {media_rec:.2f}', fontsize=11, fontweight='bold')
    plt.text(0.05, 0.33, f'Desvio Padrão: {desvio_padrao:.2f} (CV: {coef_variacao:.2f}%)', fontsize=11)
    plt.text(0.05, 0.26, f'IQR (Intervalo Interquartil): {iqr:.2f}', fontsize=11)
    plt.text(0.05, 0.19, f'Amplitude Total: {amplitude}', fontsize=11)
    plt.text(0.05, 0.12, f'Distância Média x Mediana: {distancia:.2f}%', fontsize=11)
    plt.title('Resumo Estatístico')
    plt.axis('off')

    # Subplot 3: Histograma de frequência com linhas de Média e Mediana
    plt.subplot(2, 2, 3)
    plt.hist(array_rec, bins=15, edgecolor='black', color='steelblue')
    plt.axvline(media_rec, color='green', linestyle='--', linewidth=1.5, label=f'Média ({media_rec:.0f})')
    plt.axvline(mediana_rec, color='orange', linestyle='-', linewidth=1.5, label=f'Mediana ({mediana_rec:.0f})')
    plt.title('Histograma de Frequência (Assimetria Positiva)')
    plt.xlabel('Veículos Recuperados')
    plt.ylabel('Quantidade de Delegacias')
    plt.legend()

    # Subplot 4: Gráfico de barras com as 10 delegacias que mais recuperam veículos
    plt.subplot(2, 2, 4)
    df_top10 = (
        df_rec_maiores
        .sort_values(by='recuperacao_veiculos', ascending=False)
        .head(10)
        .sort_values(by='recuperacao_veiculos', ascending=True)
    )
    rotulos = df_top10['cisp'].astype(str) + 'ª DP (' + df_top10['munic'].str[:12] + ')'
    plt.barh(rotulos, df_top10['recuperacao_veiculos'], color='darkslateblue')
    plt.title('Top 10 Delegacias com Maiores Recuperações')
    plt.xlabel('Veículos Recuperados')
    plt.ylabel('Delegacia (CISP)')

    plt.suptitle('Análise Estatística de Recuperação de Veículos por Delegacia - ISP/RJ', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('painel_recuperacao_veiculos.png', dpi=300)
    plt.show()

except Exception as e:
    print(f'Erro ao plotar os gráficos: {e}')


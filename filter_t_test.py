import pandas as pd
import re


def analizza_ttest_mirato(file_input, file_output='ttest_filtrati_ANN_vs_KNN.csv', alpha=0.05):
    print(f"Lettura del file: {file_input}")

    # Leggiamo il TSV. Assegniamo noi i nomi alle colonne
    df = pd.read_csv(file_input, sep='\t', header=None,
                     names=['Model_A', 'Model_B', 'Metric', 'p_value'])

    # Funzione avanzata per estrarre le info dal nome del modello
    def estrai_info(stringa_modello):
        # Il nome base è la prima parola prima dell'underscore (es. ItemANNOY, ItemKNNfairness, ItemKNN)
        base = stringa_modello.split('_')[0]

        # Tipo di modello (User o Item)
        tipo = 'User' if stringa_modello.startswith('User') else 'Item'

        # Estrazione del valore di nn
        match = re.search(r'nn=(\d+)', stringa_modello)
        nn_val = int(match.group(1)) if match else None

        # Determina se è la BASELINE assoluta (esattamente ItemKNN o UserKNN)
        is_baseline = (base == 'ItemKNN' or base == 'UserKNN')

        return pd.Series([base, tipo, nn_val, is_baseline])

    print("Estrazione delle informazioni dai nomi dei modelli...")
    df[['Base_A', 'Type_A', 'nn_A', 'is_baseline_A']] = df['Model_A'].apply(estrai_info)
    df[['Base_B', 'Type_B', 'nn_B', 'is_baseline_B']] = df['Model_B'].apply(estrai_info)

    # --- FILTRI ---

    # 1. Vogliamo che Model_A sia il modello da testare (ANN, FairANN, KNNfairness)
    #    e Model_B sia la BASELINE pura (ItemKNN o UserKNN).
    #    Questo elimina anche i duplicati simmetrici (B vs A).
    mask_A_is_target = ~df['is_baseline_A']
    mask_B_is_baseline = df['is_baseline_B']

    # 2. Vogliamo confrontare solo Item vs Item o User vs User
    mask_same_type = df['Type_A'] == df['Type_B']

    # 3. Vogliamo confrontare solo a parità di neighbor (nn)
    mask_same_nn = df['nn_A'] == df['nn_B']

    # Applichiamo tutti i filtri contemporaneamente
    df_filtrato = df[mask_A_is_target & mask_B_is_baseline & mask_same_type & mask_same_nn].copy()

    # --- VALUTAZIONE STATISTICA ---
    # La differenza è significativa se p-value < 0.05
    df_filtrato['Differenza_Significativa'] = df_filtrato['p_value'] < alpha

    # Pulizia delle colonne di supporto
    df_filtrato = df_filtrato.drop(columns=[
        'Base_A', 'Type_A', 'nn_A', 'is_baseline_A',
        'Base_B', 'Type_B', 'nn_B', 'is_baseline_B'
    ])

    # Ordiniamo i risultati per una lettura facilitata:
    # prima Item/User, poi Metrica, poi Modello Testato, infine Baseline
    df_filtrato = df_filtrato.sort_values(by=['Model_B', 'Metric', 'Model_A'])

    # Salvataggio
    df_filtrato.to_csv(file_output, index=False)

    print(f"\n✅ Filtraggio completato con successo!")
    print(f"Righe totali originali: {len(df)}")
    print(f"Confronti validi trovati (ANN/Fair vs Baseline, stesso nn, stesso Tipo): {len(df_filtrato)}")
    print(f"File salvato in: {file_output}")

    # Breve riepilogo
    significativi = df_filtrato['Differenza_Significativa'].sum()
    print(
        f"Di questi confronti validi, {significativi} presentano una differenza statisticamente significativa (p < {alpha}).")


# Inserisci il nome del tuo file TSV generato da Elliot
NOME_FILE_TSV = "stat_paired_ttest_cutoff_20_relthreshold_0_2026_02_28_02_15_18.tsv"
analizza_ttest_mirato(NOME_FILE_TSV)
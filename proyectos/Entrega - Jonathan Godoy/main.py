import pandas as pd 
from tqdm import tqdm
import re
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import skfuzzy as fuzz
import time

tqdm.pandas()
nltk.download('vader_lexicon')

def process_text(text:str):
    """
    Procesa el texto eliminando URLs, nombres de usuario y caracteres especiales
    Tambien reemplaza contracciones por su forma extendida.

    Args:
    text (str): Texto a procesar.
    
    Returns:
    str: Texto procesado.
    """

    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r' www\S+', '', text)
    text = re.sub(r"#", "", text)
    text = re.sub(r'@', '', text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r" t ", " not ", text)
    text = re.sub(r"it\'s", " it is", text)
    text = re.sub(r"i\'m", " i am", text)
    text = re.sub(r"\'m ", " am ", text)
    text = re.sub(r" m "," am ", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r" re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r" d ", " would ", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r" s ", " is ", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r" ll ", " will ", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r" ve ", " have ", text)
    text = re.sub(r"\'cause", " because", text)
    text = re.sub(r" cause", " because", text)
    text = re.sub(r"\'n", " and", text)
    text = re.sub(r" n ", " and ", text)

    return text

def analyze_sentiment(df):
    """
    Analiza el sentimiento de las oraciones en el DataFrame y agrega nuevas columnas.
    
    Args:
    df (pandas.DataFrame): DataFrame con columnas 'sentence' y 'sentiment' (0 para negativo, 1 para positivo).
    
    Returns:
    pandas.DataFrame: DataFrame con nuevas columnas TweetPos, TweetNeg, TweetNeu y una comparación con 'sentiment'.
    """

    # Crear una copia del DataFrame original para no modificarlo
    df_with_sentiment = df.copy() 

    # Inicializar el analizador de sentimientos de VADER
    sia = SentimentIntensityAnalyzer()

    df_with_sentiment["TweetPos"] = df_with_sentiment["sentence"].apply(lambda x: sia.polarity_scores(x)["pos"])
    df_with_sentiment["TweetNeu"] = df_with_sentiment["sentence"].apply(lambda x: sia.polarity_scores(x)["neu"])
    df_with_sentiment["TweetNeg"] = df_with_sentiment["sentence"].apply(lambda x: sia.polarity_scores(x)["neg"])
    

    return df_with_sentiment
    
def fuzzification(df):  
    """
    Fuzzifica los scores de sentimiento del DataFrame.
    
    Args:
    df (pandas.DataFrame): DataFrame con columnas TweetPos y TweetNeg.
    
    Returns:
    pandas.DataFrame: DataFrame con nuevas columnas fuzzificadas.
    """

    # Crear una copia del DataFrame original para no modificarlo
    df_fuzzified = df.copy()

    # Definir los rangos para fuzzificación 
    min_val, max_val = 0, 1
    mid_val = (min_val + max_val) / 2

    # Variables de fuzzificación para los positivos, negativos
    x_pos = np.arange(min_val, max_val, 0.1)
    x_neg = np.arange(min_val, max_val, 0.1)

    # Funciones de membresía para los positivos
    pos_low = fuzz.trimf(x_pos, [min_val, min_val, mid_val])
    pos_medium = fuzz.trimf(x_pos, [min_val, mid_val, max_val])
    pos_high = fuzz.trimf(x_pos, [mid_val, max_val, max_val])

    # Funciones de membresía para los negativos
    neg_low = fuzz.trimf(x_neg, [min_val, min_val, mid_val])
    neg_medium = fuzz.trimf(x_neg, [min_val, mid_val, max_val])
    neg_high = fuzz.trimf(x_neg, [mid_val, max_val, max_val])

    # Medir tiempo de fuzzificación por cada tweet
    df_fuzzified['fuzzification_time'] = 0.0

    for index, row in df_fuzzified.iterrows():
        start_time = time.perf_counter()
        # Fuzzificación para valores positivos
        df_fuzzified.at[index,'Pos_Low'] = fuzz.interp_membership(x_pos, pos_low, row['TweetPos'])
        df_fuzzified.at[index,'Pos_Medium'] = fuzz.interp_membership(x_pos, pos_medium, row['TweetPos'])
        df_fuzzified.at[index,'Pos_High'] = fuzz.interp_membership(x_pos, pos_high, row['TweetPos'])
        # Fuzzificación para valores negativos
        df_fuzzified.at[index,'Neg_Low'] = fuzz.interp_membership(x_neg, neg_low, row['TweetNeg'])
        df_fuzzified.at[index,'Neg_Medium'] = fuzz.interp_membership(x_neg, neg_medium, row['TweetNeg'])
        df_fuzzified.at[index,'Neg_High'] = fuzz.interp_membership(x_neg, neg_high, row['TweetNeg'])

        # Finaliza el temporizador y almacena el tiempo de fuzzificación
        end_time = time.perf_counter()
        df_fuzzified.at[index, 'fuzzification_time'] = end_time - start_time
    return df_fuzzified

def rule_evaluation(row):
    """
    Evalúa las reglas difusas para una fila del DataFrame.
    
    Args:
    row (pandas.Series): Una fila del DataFrame.
    
    Returns:
    dict: Diccionario con los pesos de activación de cada regla.
    """
    rules = {
        'w_R1': np.fmin(row['Pos_Low'], row['Neg_Low']), # R1: Pos Low AND Neg Low -> Neutral
        'w_R2': np.fmin(row['Pos_Medium'], row['Neg_Low']), # R2: Pos Medium AND Neg Low -> Positive 
        'w_R3': np.fmin(row['Pos_High'], row['Neg_Low']), # R3: Pos High AND Neg Low -> Positive
        'w_R4': np.fmin(row['Pos_Low'], row['Neg_Medium']), # R4: Pos Low AND Neg Medium -> Negative
        'w_R5': np.fmin(row['Pos_Medium'], row['Neg_Medium']), # R5: Pos Medium AND Neg Medium -> Neutral
        'w_R6': np.fmin(row['Pos_High'], row['Neg_Medium']), # R6: Pos High AND Neg Medium -> Positive
        'w_R7': np.fmin(row['Pos_Low'], row['Neg_High']), # R7: Pos Low AND Neg High -> Negative
        'w_R8': np.fmin(row['Pos_Medium'], row['Neg_High']), # R8: Pos Medium AND Neg High -> Negative
        'w_R9': np.fmin(row['Pos_High'], row['Neg_High']) # R9: Pos High AND Neg High -> Neutral
    }
    return rules    

def evaluate_rules(df):
    """
    Aplica la evaluación de reglas a todo el DataFrame.
    
    Args:
    df (pandas.DataFrame): DataFrame con columnas fuzzificadas.
    
    Returns:
    pandas.DataFrame: DataFrame con una nueva columna 'rules' conteniendo los pesos de activación.
    """
    # Crear una copia del DataFrame original para no modificarlo
    df_evaluated = df.copy()

    # Evaluar las reglas para cada fila del DataFrame
    df_evaluated['rules'] = df_evaluated.apply(rule_evaluation, axis=1)
    
    return df_evaluated

def aggregation(op_neg, op_neu, op_pos, row):
    """
    Realiza la agregación de las salidas de las reglas difusas.
    
    Args:
    op_neg, op_neu, op_pos (numpy.array): Funciones de membresía de salida.
    row (pandas.Series): Una fila del DataFrame.
    
    Returns:
    numpy.array: Salida agregada.
    """

    rules = row['rules']
    # Agregar reglas negativas (w_neg)
    w_neg = np.fmax(rules['w_R4'], np.fmax(rules['w_R7'], rules['w_R8']))

    # Agregar reglas neutras (w_neu)
    w_neu = np.fmax(rules['w_R1'], np.fmax(rules['w_R5'], rules['w_R9']))

    # Agregar reglas positivas (w_pos)
    w_pos = np.fmax(rules['w_R2'], np.fmax(rules['w_R3'], rules['w_R6']))
    
    # Aplicar las funciones de salida a los pesos agregados
    op_activation_low = np.fmin(w_neg, op_neg)
    op_activation_med = np.fmin(w_neu, op_neu)
    op_activation_high = np.fmin(w_pos, op_pos)

    # Agregar las salidas (union de las funciones de membresía)
    aggregated_output = np.fmax(op_activation_low, np.fmax(op_activation_med, op_activation_high))

    return aggregated_output

def aggregate_rule_outputs(df):
    """
    Aplica la agregación de reglas a todo el DataFrame.
    
    Args:
    df (pandas.DataFrame): DataFrame con la columna 'rules'.
    
    Returns:
    pandas.DataFrame: DataFrame con una nueva columna 'aggregated_output'.
    """
    
    # Crear una copia del DataFrame original para no modificarlo
    df_aggregated = df.copy()

    # Definir funciones de salida (MFs)
    x_op = np.arange(0, 10, 1)
    op_neg = fuzz.trimf(x_op, [0, 0, 5])
    op_neu = fuzz.trimf(x_op, [0, 5, 10])
    op_pos = fuzz.trimf(x_op, [5, 10, 10])

    # Aplicar agregación de reglas para cada fila del DataFrame
    df_aggregated['aggregated_output'] = df_aggregated.apply(lambda row: aggregation(op_neg, op_neu, op_pos, row), axis=1)
    return df_aggregated, x_op

def defuzzification(df, x_op):
    """
    Realiza la defuzzificación del output agregado y clasifica el sentimiento.
    
    Args:
    df (pandas.DataFrame): DataFrame con la columna 'aggregated_output'.
    
    Returns:
    pandas.DataFrame: DataFrame con una nueva columna 'sentiment_classification'.
    """

    # Crear una copia del DataFrame original para no modificarlo
    df_defuzzified = df.copy()

    # Definir el rango de la variable de salida 
    max_val, min_val = 10, 0
    z_output = x_op

    # Medir tiempo de defuzzificación por cada tweet
    df_defuzzified['defuzzification_time'] = 0.0

    # Para cada fila del DataFrame, aplicar el proceso de defuzzificación
    for index, row in df_defuzzified.iterrows():
        # Definir el rango de la variable de salida 
        start_time = time.perf_counter()

        # Defuzzificar el output agregado con el método del centroide
        COA = fuzz.defuzz(z_output, row['aggregated_output'], 'centroid')
          
        # Clasificar el sentimiento basado en el valor del COA
        if min_val <= COA < max_val/3:
            df_defuzzified.at[index, 'sentiment_classification'] = 'Negative'
        elif max_val/3 <= COA < 2*max_val/3:
            df_defuzzified.at[index, 'sentiment_classification'] = 'Neutral'
        else:
            df_defuzzified.at[index, 'sentiment_classification'] = 'Positive'

        # Finaliza el temporizador y almacena el tiempo de defuzzificación
        end_time = time.perf_counter()
        df_defuzzified.at[index, 'defuzzification_time'] = end_time - start_time
        df_defuzzified.at[index, 'COA'] = COA

    return df_defuzzified

def benchmarks(df):
    total_tweets_positive = df[df['sentiment_classification'] == 'Positive'].shape[0]
    total_tweets_neutral = df[df['sentiment_classification'] == 'Neutral'].shape[0]
    total_tweets_negative = df[df['sentiment_classification'] == 'Negative'].shape[0]

    total_time_fuzzification_positive = df[df['sentiment_classification'] == 'Positive']['fuzzification_time'].sum()
    total_time_fuzzification_neutral = df[df['sentiment_classification'] == 'Neutral']['fuzzification_time'].sum()
    total_time_fuzzification_negative = df[df['sentiment_classification'] == 'Negative']['fuzzification_time'].sum()

    total_time_defuzzification_positive = df[df['sentiment_classification'] == 'Positive']['defuzzification_time'].sum()
    total_time_defuzzification_neutral = df[df['sentiment_classification'] == 'Neutral']['defuzzification_time'].sum()
    total_time_defuzzification_negative = df[df['sentiment_classification'] == 'Negative']['defuzzification_time'].sum()

    # Imprimir los resultados de los tweets
    print("\n+"+"-"*107+"+---------------+----------+-----------+------------+---------+")
    print("|"+" "*45+"10 primeros Tweets"+" "*44+"| Clasificación |    Pos   |     Neg   |     Neu    |   COA   |")
    print("+"+"-"*107+"+---------------+----------+-----------+------------+---------+")
    
    for _, row in df.head(10).iterrows():
        tweet = row['sentence'][:102] + '...' if len(row['sentence']) > 102 else row['sentence']
        classification = row['sentiment_classification']
        tweet_pos = f"{row['TweetPos']:.4f}"
        tweet_neu = f"{row['TweetNeu']:.4f}"
        tweet_neg = f"{row['TweetNeg']:.4f}"
        coa = f"{row['COA']:.4f}"
        
        print(f"| {tweet:<105} | {classification:<13} | {tweet_pos:>8} | {tweet_neg:>9} | {tweet_neu:>10} | {coa:>7} |")
    
    print("+"+"-"*107+"+---------------+----------+-----------+------------+---------+")
    
    benchmark_results = {
        'total_tweets_positive': total_tweets_positive,
        'total_tweets_neutral': total_tweets_neutral,
        'total_tweets_negative': total_tweets_negative,
        'total_time_fuzzification_positive': total_time_fuzzification_positive,
        'total_time_fuzzification_neutral': total_time_fuzzification_neutral,
        'total_time_fuzzification_negative': total_time_fuzzification_negative,
        'total_time_defuzzification_positive': total_time_defuzzification_positive,
        'total_time_defuzzification_neutral': total_time_defuzzification_neutral,
        'total_time_defuzzification_negative': total_time_defuzzification_negative
    }

    return benchmark_results

if __name__ == "__main__":
    # Cargar el dataset (MODULO 1)
    data_dir =  "./data/test_data.csv"
    df = pd.read_csv(data_dir)

    # Iniciar el cronómetro para medir el tiempo de ejecución total
    start_total = time.time()

    # Procesar el texto
    print('\nProcesando texto...')
    df['sentence'] = df['sentence'].progress_map(process_text)
    if df is not None:
        # Analizar el sentimiento de las oraciones (MODULO 2)
        df = analyze_sentiment(df)

        # Fuzzificar los puntajes de sentimiento (MODULO 3)
        df= fuzzification(df)
 
        # Evaluar las reglas (MODULO 4)
        df = evaluate_rules(df)

        # Agregar las salidas de las reglas
        df, x_op = aggregate_rule_outputs(df)

        # Defuzzificar para obtener la clasificación final (MODULO 5)
        df = defuzzification(df,x_op)

        # Calcular el tiempo de ejecución para cada tweet
        time_execution = df['fuzzification_time'] + df['defuzzification_time']
        df['execution_time'] = time_execution.apply(lambda x: f"{x:.6f}")

        # Finalizar el cronómetro para medir el tiempo de ejecución total
        total_execution_time = time.time() - start_total

        # Generar los benchmarks (MODULO 6)
        benchmark_results = benchmarks(df)

        # Mostrar Resultados de Benchmarks por consola
        print("\n")
        print("\n+--------------------------------------------------------+------------+")
        print(f"| Total de tweets positivos:                             |      {benchmark_results['total_tweets_positive']:<5} |")
        print(f"| Total de tweets neutrales:                             |      {benchmark_results['total_tweets_neutral']:<5} |")
        print(f"| Total de tweets negativos:                             |      {benchmark_results['total_tweets_negative']:<5} |")
        print(f"| Total de tweets:                                       |      {df.shape[0]:<5} |")
        print(f"| Tiempo total de ejecución:                             | {total_execution_time:.4f} seg |")
        print(f"| Tiempo total de fuzzificación para tweets positivos:   | {benchmark_results['total_time_fuzzification_positive']:.4f} seg |")
        print(f"| Tiempo total de fuzzificación para tweets neutrales:   | {benchmark_results['total_time_fuzzification_neutral']:.4f} seg |")
        print(f"| Tiempo total de fuzzificación para tweets negativos:   | {benchmark_results['total_time_fuzzification_negative']:.4f} seg |")
        print(f"| Tiempo total de defuzzificación para tweets positivos: | {benchmark_results['total_time_defuzzification_positive']:.4f} seg |")
        print(f"| Tiempo total de defuzzificación para tweets neutrales: | {benchmark_results['total_time_defuzzification_neutral']:.4f} seg |")
        print(f"| Tiempo total de defuzzificación para tweets negativos: | {benchmark_results['total_time_defuzzification_negative']:.4f} seg |")
        print("+--------------------------------------------------------+------------+")
        
        # Guardar los resultados en un archivo CSV
        df['fuzzification_time'] = df['fuzzification_time'].apply(lambda x: f"{x:.6f}")
        df['defuzzification_time'] = df['defuzzification_time'].apply(lambda x: f"{x:.6f}")
        df['COA'] = df['COA'].apply(lambda x: f"{x:.6f}".replace('.',','))

        df = df.rename(columns={
            'sentence': 'Tweet',
            'sentiment': 'Label',
            'TweetPos': 'Puntaje Positivo',
            'TweetNeg': 'Puntaje Negativo',
            'TweetNeu': 'Puntaje Neutro',
            'sentiment_classification': 'Clasificacion',
            'execution_time': 'Tiempo de ejecucion Total',
            'fuzzification_time': 'Tiempo de Fuzzificacion',
            'defuzzification_time': 'Tiempo de Defuzzificacion',
        })
        df[['Tweet', 'Label', 'Puntaje Positivo', 'Puntaje Negativo', 'Puntaje Neutro', 'Clasificacion', 'COA','Tiempo de Fuzzificacion', 'Tiempo de Defuzzificacion', 'Tiempo de ejecucion Total']].to_csv('./output/output.csv', index=False, sep=';')

        #import os
        #os.system("start ./output/output.csv")





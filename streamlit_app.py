import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import io
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import math
from typing import List, Tuple

st.title('💻 Кластеризация на основе файлов эксель lalalalala')

st.info("Это веб-приложение для кластеризации ваших данных, хранящихся в эксель-файлах")

with st.expander('Импорт данных'):

  unploaded_file = st.file_uploader(label="Загрузите свой файл")
  session_state.preparation_state = False
    
with st.expander('Подготовка датасета'):
  
  if unploaded_file:
    st.header("Введите параметры подготовки данных")
    col_index_change = st.selectbox("Выберите вариант индексирования", ("В датасете нет колонки для индекса", "Индексом датасета является первый столбец"))

    null_transform = st.selectbox("Выберите вариант обработки пустых значений переменных", ("Удалять строки, содержащие пустые значения", "Заменять пустые значения на моду в колонке"))

    categorial_to_numerical = st.selectbox("Выберите вариант преобразования категориальных переменных в численные", ("OrdinalEncoder", "OneHotEncoder"))

    scaler_method = st.selectbox("Выберите вариант нормализации данных", ("Не производить нормализацию", "Стандартизация (StandartScaler)", "Масштабирование с помощью MinMaxScaler", "Масштабирование с помощью RobustScaler"))

    def preparation_state_button_on_click():
      if col_index_change == "В датасете нет колонки для индекса":
        df = pd.read_excel(unploaded_file)
      else:
        df = pd.read_excel(unploaded_file, index_col = 0)

      df.dropna(axis=1, how='all', inplace=True)

      if null_transform == "Удалять строки, содержащие пустые значения":
        df = df.dropna()
      else: 
        for col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

      if categorial_to_numerical == "OrdinalEncoder":
        df = OrdinalEncoder().fit_transform(df)
      else:
        df = OneHotEncoder().fit_transform(df)
      
      if scaler_method != "Не производить нормализацию"
        if scaler_method == "Стандартизация (StandartScaler)":
          scaler = StandardScaler()
        else if scaler_method == "Масштабирование с помощью MinMaxScaler":
          scaler = MinMaxScaler()
        else if scaler_method == "Масштабирование с помощью RobustScaler"
          scaler = RobustScaler()
        df = scaler.fit_transform(df)

      session_state.preparation_state = True
      return df
    
    preparation_state_button = st.button("Провести предобработку", on_click=preparation_state_button_on_click)

  else:
    st.write('Загрузите файл во вкладке "Импорт данных"')

with st.expander('Кластеризация методом k-means++'):
      
  if unploaded_file:
    if session_state.preparation_state:
      if df.shape[0]>=3:
        elbow_method_need = st.selectbox("Требуется ли построить график локтя для лучшего представления о необходимом количестве кластеров?", ("Нет", "Да"), key="elbow_method_need_box")
        
        if elbow_method_need=="Да":
          
          if df.shape[0]<=100:
            clusters_quan_elbow_method = st.selectbox("Укажите максимальное количество кластеров",["Не выбрано"]+[i for i in range (3,df.shape[0]+1)], key = "clusters_quan_elbow_method_key")
          else:
            clusters_quan_elbow_method = st.selectbox("Укажите максимальное количество кластеров",["Не выбрано"]+[i for i in range (3,100)], key = "clusters_quan_elbow_method_key")
      
          def elbow_method(df, max_clusters_quan):    
            # st.session_state.clicked = True
            ssd = []
            scaler = StandardScaler()
            df = scaler.fit_transform(df)
            for quan_of_clusters in range(2, max_clusters_quan+1):
                model = KMeans(n_clusters=quan_of_clusters, init="k-means++")
                model.fit(df)
                ssd.append(model.inertia_)
            plt.plot(range(2, max_clusters_quan+1), ssd, "o--")
            plt.title("График локтя")
            plt.xlabel("Количество кластеров")
            plt.ylabel("SSD")
            # Get the current axes
            ax = plt.gca()
            # Set x-axis to only display integers
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # st.session_state["elbow_plot"] = st.pyplot(plt)
            return st.pyplot(plt)
  
  
  
          # st.session_state["elbow_method_button_clicked"] 
          # def click_button():
            # myplot = elbow_method(df, clusters_quan_elbow_method)
          #   st.session_state["elbow_method_button_clicked"] = True
            # elbow_method(df, clusters_quan_elbow_method)
          
          if clusters_quan_elbow_method!="Не выбрано":
            elbow_method_button = st.button("Построить график локтя")
  
            if elbow_method_button:
              elbow_method(df, clusters_quan_elbow_method)
  
        if df.shape[0]<=100:
          k_means_cluster_quan = st.selectbox("Укажите количество кластеров",["Не выбрано"]+[i for i in range (3,df.shape[0]+1)], key="clusters_quan_k_plus_plus")
        else:
          k_means_cluster_quan = st.selectbox("Укажите количество кластеров",["Не выбрано"]+[i for i in range (3,100)], key="clusters_quan_k_plus_plus")
      
            
        def k_means_plus_plus(df, quan_of_clusters):
          try:
            scaler = StandardScaler()
            df = scaler.fit_transform(df)
            model = KMeans(n_clusters = quan_of_clusters, init = "k-means++")
            cluster_labels = model.fit_predict(df)
            df["Номер кластера"] = cluster_labels
            st.session_state["current_df"] = df
            return df
                           
          except Exception as e:
            st.write(f"Ошибка при кластеризации {e}")
            return None
            
        if k_means_cluster_quan!="Не выбрано": 
          df = k_means_plus_plus(df, k_means_cluster_quan)
          st.session_state["current_df"]
        # Create a Pandas Excel writer using XlsxWriter as the engine.
          buffer = io.BytesIO()
          
          with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
              # Write each dataframe to a different worksheet.
              st.session_state["current_df"].to_excel(writer, sheet_name='k_means')
          
              # Close the Pandas Excel writer and output the Excel file to the buffer
              writer.close()
          
              st.download_button(
                  label="Загрузить датафрейм в эксель-файл",
                  data=buffer,
                  file_name="dataframe_k_means_algorithm.xlsx",
                  mime="application/vnd.ms-excel"
              )
  
      
              
      else:
        st.write("В датасете меньше трёх строк, кластеризация бессмысленна. Увеличьте количество строк или измените параметры подгтовки датасета, если в исходном датасете строк больше")
    else:
      st.write("Проведите предобработку загруженных данных")
  else:
    st.write('Загрузите файл во вкладке "Импорт данных" и проведите предобработку загруженных данных')
    
with st.expander('Иерархическая кластеризация'):  
  if unploaded_file:
    if session_state.preparation_state:
      if df.shape[0]>=3:
        st.write("Я здеся!")
  
        def hierarchy_dendrogram(df, level=31):
          scaler = MinMaxScaler()
          scaled_data = scaler.fit_transform(df)
          df = pd.DataFrame(scaled_data, columns=df.columns)
          linkage_matrix = hierarchy.linkage(df.values, method="ward")
          # Create a figure and axis for the plot
          fig, ax = plt.subplots(figsize=(20, 10), dpi=200)
          ax.set_title("Дендрограмма", fontsize=30)
          
          dendrogram(linkage_matrix, truncate_mode="level", p=level-1, ax=ax)
          
          st.pyplot(fig)
  
          return None
  
        dendrogram_need = st.selectbox("Требуется ли построить дендрограмму для лучшего представления о необходимом количестве кластеров?", ("Нет", "Да"), key="dendrogram_need_box")
        
        if dendrogram_need=="Да":
          if df.shape[0]<=100:
            dendrogram_level = st.selectbox("Выберите уровень глубины дендрограммы",["Не выбрано"]+[i for i in range (3,df.shape[0]+1)])
          else:
            dendrogram_level = st.selectbox("Выберите уровень глубины дендрограммы",["Не выбрано"]+[i for i in range (3,100)])
            
          if dendrogram_level!="Не выбрано":
            hierarchy_dendrogram(df, int(dendrogram_level))
             
        if df.shape[0]<=100:
          hierarchy_cluster_quan = st.selectbox("Укажите количество кластеров",["Не выбрано"]+[i for i in range(3, df.shape[0]+1)], key="clusters_quan_hierarchy")
          # hierarchy_cluster_quan = st.selectbox("Укажите количество кластеров",["Не выбрано"]+[i for i in range (3,df.shape[0]+1)], )
        else:
          hierarchy_cluster_quan = st.selectbox("Укажите количество кластеров",["Не выбрано"]+[i for i in range (3,100)], key="clusters_quan_hierarchy")
          
        def hierarchy_clusterisation(df, quan_of_clusters):
          scaler = MinMaxScaler()
          scaled_data = scaler.fit_transform(df)
          df = pd.DataFrame(scaled_data, columns=df.columns)
          model = AgglomerativeClustering(quan_of_clusters)
          cluster_labels = model.fit_predict(df)
          df["Номер кластера"] = cluster_labels
          st.session_state["current_df"] = df
          return df
  
        if hierarchy_cluster_quan!="Не выбрано": 
          df = hierarchy_clusterisation(df, hierarchy_cluster_quan)
          st.session_state["current_df"]
        # Create a Pandas Excel writer using XlsxWriter as the engine.
          buffer = io.BytesIO()
          
          with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
              # Write each dataframe to a different worksheet.
              st.session_state["current_df"].to_excel(writer, sheet_name='k_means')
          
              # Close the Pandas Excel writer and output the Excel file to the buffer
              writer.close()
          
              st.download_button(
                  label="Загрузить датафрейм в эксель-файл",
                  data=buffer,
                  file_name="dataframe_hierarchy_algorithm.xlsx",
                  mime="application/vnd.ms-excel"
              )
      
      else:
        st.write("В датасете меньше трёх строк, кластеризация бессмысленна. Увеличьте количество строк или измените параметры подгтовки датасета, если в исходном датасете строк больше")
    else:
      st.write("Проведите предобработку загруженных данных")
  else:
    st.write('Загрузите файл во вкладке "Импорт данных" и проведите предобработку загруженных данных')

with st.expander('Метод DBSCAN'):  
  if unploaded_file:
    if session_state.preparation_state:
      if df.shape[0]>=3:
        st.write("luala")
        scaler = MinMaxScaler()
        df = scaler.fit_transform(df)
  
        eps_to_use = st.number_input("Выберите параметр эпсилон", value=0.01)
        min_samples_to_use = st.selectbox("Выберите параметр min_samples", [i for i in range(len(df)+1)])
      
      else:
        st.write("В датасете меньше трёх строк, кластеризация бессмысленна. Увеличьте количество строк или измените параметры подгтовки датасета, если в исходном датасете строк больше")
    else:
      st.write("Проведите предобработку загруженных данных")
  else:
    st.write('Загрузите файл во вкладке "Импорт данных" и проведите предобработку загруженных данных')

  

    




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
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

st.title('💻 Кластеризация на основе файлов эксель lalalal')

st.info("Это веб-приложение для кластеризации ваших данных, хранящихся в эксель-файлах")

with st.expander('Импорт данных'):
  
  unploaded_file = st.file_uploader(label="Загрузите свой файл")
  unploaded_file_df = pd.read_excel(unploaded_file)
  unploaded_file_df
  
    
with st.expander('Подготовка датасета'):
  
  if unploaded_file:
    st.header("Введите параметры подготовки данных")
    col_index_change = st.selectbox("Выберите вариант индексирования", ("В датасете нет колонки для индекса", "Индексом датасета является первый столбец"))

    null_transform = st.selectbox("Выберите вариант обработки пустых значений переменных", ("Удалять строки, содержащие пустые значения", "Заменять пустые значения на среднее для численных и моду для категориальных переменных в колонке"))

    categorial_to_numerical = st.selectbox("Выберите вариант преобразования категориальных переменных в численные", ("OrdinalEncoder", "OneHotEncoder"))

    scaler_method = st.selectbox("Выберите вариант нормализации данных", ("Не производить нормализацию", "Стандартизация (StandartScaler)", "Масштабирование с помощью MinMaxScaler", "Масштабирование с помощью RobustScaler"))

    def st.session_state.preparation_state_button_on_click():
      st.write("yf;fkb ryjgre!")
    
      # if col_index_change == "В датасете нет колонки для индекса":
      #   df = pd.read_excel(unploaded_file)
      # else:
      #   df = pd.read_excel(unploaded_file, index_col = 0)

      # # df.dropna(axis=1, how='all', inplace=True)

      # # if null_transform == "Удалять строки, содержащие пустые значения":
      # #   df = df.dropna()
      # # else: 
      # #   for col in df.columns:
      # #     for el in df[col]:
      # #       if not pd.notna(el):
      # #         if str(el).replace(".", "", 1).isdigit():
      # #           df[col].fillna(df[col].mean()[0], inplace=True)
      # #           break
      # #         else:
      # #           df[col].fillna(df[col].mode()[0], inplace=True)
      # #           break

      # # if categorial_to_numerical == "OrdinalEncoder":
      # #   df = OrdinalEncoder().fit_transform(df)
      # # else:
      # #   df = OneHotEncoder().fit_transform(df)
      
      # # if scaler_method != "Не производить нормализацию":
      # #   if scaler_method == "Стандартизация (StandartScaler)":
      # #     scaler = StandardScaler()
      # #   elif scaler_method == "Масштабирование с помощью MinMaxScaler":
      # #     scaler = MinMaxScaler()
      # #   elif scaler_method == "Масштабирование с помощью RobustScaler":
      # #     scaler = RobustScaler()
      # #   df = scaler.fit_transform(df)

      st.session_state.preparation_state = True
      # return None
    
    preparation_state_button = st.button("Провести предобработку", on_click=st.session_state.preparation_state_button_on_click)
    if st.session_state.preparation_state:
      df

  else:
    st.write('Загрузите файл во вкладке "Импорт данных"')

with st.expander('Кластеризация методом k-means++'):  
  if unploaded_file:
    if st.session_state.preparation_state:
      if df.shape[0]>=3:
        k_means_df = df
        elbow_method_need = st.selectbox("Требуется ли построить график локтя для лучшего представления о необходимом количестве кластеров?", ("Нет", "Да"), key="elbow_method_need_box")
        
        if elbow_method_need=="Да":
          
          if k_means_df.shape[0]<=100:
            clusters_quan_elbow_method = st.selectbox("Укажите максимальное количество кластеров",["Не выбрано"]+[i for i in range (3,df.shape[0]+1)], key = "clusters_quan_elbow_method_key")
          else:
            clusters_quan_elbow_method = st.selectbox("Укажите максимальное количество кластеров",["Не выбрано"]+[i for i in range (3,100)], key = "clusters_quan_elbow_method_key")
      
          def elbow_method(k_means_df, max_clusters_quan):    
            ssd = []
            for quan_of_clusters in range(2, max_clusters_quan+1):
                model = KMeans(n_clusters=quan_of_clusters, init="k-means++")
                model.fit(k_means_df)
                ssd.append(model.inertia_)
            plt.plot(range(2, max_clusters_quan+1), ssd, "o--")
            plt.title("График локтя")
            plt.xlabel("Количество кластеров")
            plt.ylabel("SSD")
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            return st.pyplot(plt)

          def elbow_method_button_on_click():
            elbow_method(k_means_df, clusters_quan_elbow_method)
          
          if clusters_quan_elbow_method!="Не выбрано":
            elbow_method_button = st.button("Построить график локтя", on_click=elbow_method_button_on_click())
  
            # if elbow_method_button:
            #   elbow_method(k_means_df, clusters_quan_elbow_method)
  
        if k_means_df.shape[0]<=100:
          k_means_cluster_quan = st.selectbox("Укажите количество кластеров",["Не выбрано"]+[i for i in range (3,k_means_df.shape[0]+1)], key="clusters_quan_k_plus_plus")
        else:
          k_means_cluster_quan = st.selectbox("Укажите количество кластеров",["Не выбрано"]+[i for i in range (3,100)], key="clusters_quan_k_plus_plus")
      
            
        def k_means_plus_plus(k_means_df, quan_of_clusters):
          try:
            model = KMeans(n_clusters = quan_of_clusters, init = "k-means++")
            cluster_labels = model.fit_predict(k_means_df)
            k_means_df["Номер кластера"] = cluster_labels
            st.session_state["current_k_means_df"] = k_means_df
            return k_means_df
                           
          except Exception as e:
            st.write(f"Ошибка при кластеризации {e}")
            return None
            
        if k_means_cluster_quan!="Не выбрано": 
          k_means_df = k_means_plus_plus(k_means_df, k_means_cluster_quan)
          st.session_state["current_k_means_df"]
        # Create a Pandas Excel writer using XlsxWriter as the engine.
          buffer = io.BytesIO()
          
          with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
              # Write each dataframe to a different worksheet.
              st.session_state["current_k_means_df"].to_excel(writer, sheet_name='k_means')
          
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
    if st.session_state.preparation_state:
      if df.shape[0]>=3:
        hierarchichal_df = df
  
        def hierarchy_dendrogram(hierarchichal_df, level=31):
          hierarchichal_df = pd.DataFrame(scaled_data, columns=hierarchichal_df.columns)
          linkage_matrix = hierarchy.linkage(hierarchichal_df.values, method="ward")
          fig, ax = plt.subplots(figsize=(20, 10), dpi=200)
          ax.set_title("Дендрограмма", fontsize=30)
          dendrogram(linkage_matrix, truncate_mode="level", p=level-1, ax=ax)
          st.pyplot(fig)
          return None
  
        dendrogram_need = st.selectbox("Требуется ли построить дендрограмму для лучшего представления о необходимом количестве кластеров?", ("Нет", "Да"), key="dendrogram_need_box")
        
        if dendrogram_need=="Да":
          if hierarchichal_hierarchichal_df.shape[0]<=100:
            dendrogram_level = st.selectbox("Выберите уровень глубины дендрограммы",["Не выбрано"]+[i for i in range (3,hierarchichal_df.shape[0]+1)])
          else:
            dendrogram_level = st.selectbox("Выберите уровень глубины дендрограммы",["Не выбрано"]+[i for i in range (3,100)])

          def hierarchy_dendrogram_button_on_click():
            hierarchy_dendrogram(hierarchichal_df, int(dendrogram_level))
            
          if dendrogram_level!="Не выбрано":
            hierarchy_dendrogram_button = st.button("Построить дендрограмму", on_click=hierarchy_dendrogram_button_on_click())
            # hierarchy_dendrogram(hierarchichal_df, int(dendrogram_level))
             
        if hierarchichal_df.shape[0]<=100:
          hierarchy_cluster_quan = st.selectbox("Укажите количество кластеров",["Не выбрано"]+[i for i in range(3, hierarchichal_df.shape[0]+1)], key="clusters_quan_hierarchy")
          # hierarchy_cluster_quan = st.selectbox("Укажите количество кластеров",["Не выбрано"]+[i for i in range (3,hierarchichal_df.shape[0]+1)], )
        else:
          hierarchy_cluster_quan = st.selectbox("Укажите количество кластеров",["Не выбрано"]+[i for i in range (3,100)], key="clusters_quan_hierarchy")
          
        def hierarchy_clusterisation(hierarchichal_df, quan_of_clusters):
          hierarchichal_df = pd.DataFrame(scaled_data, columns=hierarchichal_df.columns)
          model = AgglomerativeClustering(quan_of_clusters)
          cluster_labels = model.fit_predict(hierarchichal_df)
          hierarchichal_df["Номер кластера"] = cluster_labels
          st.session_state["current_hierarchichal_df"] = hierarchichal_df
          return hierarchichal_df
  
        if hierarchy_cluster_quan!="Не выбрано": 
          hierarchichal_df = hierarchy_clusterisation(hierarchichal_df, hierarchy_cluster_quan)
          st.session_state["current_hierarchichal_df"]
        # Create a Pandas Excel writer using XlsxWriter as the engine.
          buffer = io.BytesIO()
          
          with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
              # Write each dataframe to a different worksheet.
              st.session_state["current_hierarchichal_df"].to_excel(writer, sheet_name='k_means')
          
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
    if st.session_state.preparation_state:
      if df.shape[0]>=3:
        dbscan_df = df
        st.write("luala")
  
        eps_to_use = st.number_input("Выберите параметр эпсилон", value=0.01)
        min_samples_to_use = st.selectbox("Выберите параметр min_samples", [i for i in range(len(df)+1)])
      
      else:
        st.write("В датасете меньше трёх строк, кластеризация бессмысленна. Увеличьте количество строк или измените параметры подгтовки датасета, если в исходном датасете строк больше")
    else:
      st.write("Проведите предобработку загруженных данных")
  else:
    st.write('Загрузите файл во вкладке "Импорт данных" и проведите предобработку загруженных данных')

  

    




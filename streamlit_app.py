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
import category_encoders as ce

st.title('💻 Кластеризация на основе данных из эксель-файлов')

st.info('Веб-приложение для кластеризации данных, хранящихся в эксель-файлах')

with st.expander('Импорт и предобработка данных'):
  st.write("Перед импортом убедитесь, пожалуйста, что Ваши данные представляют собой структурированную таблицу, а не просто набор данных. В противном случае это приведёт к неправильной работе программы.")
  
  unploaded_file = st.file_uploader(label="Загрузите свой файл")

  df_state = False
  if unploaded_file:
    unploaded_file_df = pd.read_excel(unploaded_file)
    unploaded_file_df

  def df_state_button_click():
    if st.session_state.button:
        df = pd.read_excel(unploaded_file)

  
  
  if unploaded_file:
   
    st.header("Введите параметры подготовки данных")
    col_index_change = st.selectbox("Выберите вариант индексирования", ("В датасете нет колонки для индекса", "Индексом датасета является первый столбец"))

    null_transform = st.selectbox("Выберите вариант обработки пустых значений переменных", ("Удалять строки, содержащие пустые значения", 
                                                                                            "Заменять пустые значения на среднее в колонке для численных и моду в колонке для категориальных переменных",
                                                                                           "Заменять пустые значения на медиану в колонке для численных и моду в колонке для категориальных переменных",
                                                                                           "Заменять пустые значения на моду в колонке для численных и категориальных переменных"))

    categorial_to_numerical = st.selectbox("Выберите вариант преобразования категориальных переменных в численные", ("OrdinalEncoder", "OneHotEncoder", "BinaryEncoder"))

    scaler_method = st.selectbox("Выберите вариант нормализации данных", ("Не производить нормализацию", "Стандартизация (StandartScaler)", "Масштабирование с помощью MinMaxScaler", "Масштабирование с помощью RobustScaler"))

    if 'preparation_state_button_clicked' not in st.session_state:
      st.session_state.preparation_state_button_clicked = False

    def preparation_state_button_on_click():
        st.session_state.preparation_state_button_clicked = True
    
    preparation_state_button = st.button("Провести предобработку", on_click=preparation_state_button_on_click)


    
    if st.session_state.preparation_state_button_clicked:
        if col_index_change != "Индексом датасета является первый столбец":
          df = unploaded_file_df.copy()
        else:
          df = pd.read_excel(unploaded_file, index_col = 0)
          # df = unploaded_file_df.copy()

        df.dropna(axis=1, how='all', inplace=True)

        if null_transform == "Удалять строки, содержащие пустые значения":
          df.dropna(axis=0, how="any", inplace=True)
        elif null_transform == "Заменять пустые значения на среднее в колонке для численных и моду в колонке для категориальных переменных": 
          df_filled = df.copy()
          for column in df_filled.columns:
              if df_filled[column].isnull().any():
                  if pd.api.types.is_numeric_dtype(df_filled[column]):
                      mean_value = df_filled[column].mean()
                      df_filled[column].fillna(mean_value, inplace=True)
                  else:
                      mode_value = df_filled[column].mode()
                      if not mode_value.empty:
                          df_filled[column].fillna(mode_value[0], inplace=True)
          df = df_filled
        elif null_transform == "Заменять пустые значения на медиану в колонке для численных и моду в колонке для категориальных переменных":
          df_filled = df.copy()
          for column in df_filled.columns:
              if df_filled[column].isnull().any():
                  if pd.api.types.is_numeric_dtype(df_filled[column]):
                      median_value = df_filled[column].median()
                      df_filled[column].fillna(median_value, inplace=True)
                  else:
                      mode_value = df_filled[column].mode()
                      if not mode_value.empty:
                          df_filled[column].fillna(mode_value[0], inplace=True)
          df = df_filled
        else:
          df_filled = df.copy()
          for column in df_filled.columns:
             mode_value = df_filled[column].mode()
             if not mode_value.empty:
                df_filled[column].fillna(mode_value[0], inplace=True)
          df = df_filled
          
        columns_to_encode = []    
        for column in df.columns:
          if not pd.api.types.is_numeric_dtype(df[column]):
            columns_to_encode.append(column)

        if categorial_to_numerical == "OrdinalEncoder":
          encoder = OrdinalEncoder()
          df[columns_to_encode] = encoder.fit_transform(df[columns_to_encode])
        elif categorial_to_numerical == "OneHotEncoder":
          ohe = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
          ohetransform = ohe.fit_transform(df[columns_to_encode])
          df = pd.concat([df, ohetransform], axis=1).drop(columns=columns_to_encode)
        else:
          be = ce.BinaryEncoder(cols=columns_to_encode, return_df=True)
          be_transform = be.fit_transform(df)
          df = be_transform

        
      
        if scaler_method != "Не производить нормализацию":
          if scaler_method == "Стандартизация (StandartScaler)":
            scaler = StandardScaler()
          elif scaler_method == "Масштабирование с помощью MinMaxScaler":
            scaler = MinMaxScaler()
          elif scaler_method == "Масштабирование с помощью RobustScaler":
            scaler = RobustScaler()
          # df = scaler.fit_transform(df)
          scaled_data = scaler.fit_transform(df)
          df = pd.DataFrame(scaled_data, columns=list(df.columns))
  
        df_state = True
        df
          

with st.expander('Кластеризация методом k-means'):  
  if unploaded_file:
    if df_state:
      if df.shape[0]>=3:
        k_means_df = df.copy()
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

          if 'elbow_method_button_clicked' not in st.session_state:
              st.session_state.elbow_method_button_clicked = False

          def elbow_method_button_on_click():
              st.session_state.elbow_method_button_clicked = True
          
          if clusters_quan_elbow_method!="Не выбрано":
            # elbow_method_button = st.button("Построить график локтя", on_click=elbow_method_button_on_click())
            elbow_method(k_means_df, clusters_quan_elbow_method)
  
        if k_means_df.shape[0]<=100:
          k_means_cluster_quan = st.selectbox("Укажите количество кластеров",["Не выбрано"]+[i for i in range (3,k_means_df.shape[0]+1)], key="clusters_quan_k_plus_plus")
        else:
          k_means_cluster_quan = st.selectbox("Укажите количество кластеров",["Не выбрано"]+[i for i in range (3,100)], key="clusters_quan_k_plus_plus")
      
            
        def k_means_plus_plus(k_means_df, quan_of_clusters):
          model = KMeans(n_clusters = quan_of_clusters, init = "k-means++")
          cluster_labels = model.fit_predict(k_means_df)
          k_means_df["Номер кластера"] = cluster_labels
          st.session_state["current_k_means_df"] = k_means_df
          return k_means_df

            
        if k_means_cluster_quan!="Не выбрано": 
          k_means_df = k_means_plus_plus(k_means_df, k_means_cluster_quan)
          st.session_state["current_k_means_df"]
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
    st.write('Загрузите файл и проведите предобработку загруженных данных во вкладке "Импорт и предобработка данных"')
    
with st.expander('Иерархическая кластеризация'):  
  if unploaded_file:
    if df_state:
      if df.shape[0]>=3:
        hierarchichal_df = df.copy()
  
        def hierarchy_dendrogram(hierarchichal_df, level=31):
          linkage_matrix = hierarchy.linkage(hierarchichal_df.values, method="ward")
          fig, ax = plt.subplots(figsize=(20, 10), dpi=200)
          ax.set_title("Дендрограмма", fontsize=30)
          dendrogram(linkage_matrix, truncate_mode="level", p=level-1, ax=ax)
          st.pyplot(fig)
          return None
  
        dendrogram_need = st.selectbox("Требуется ли построить дендрограмму для лучшего представления о необходимом количестве кластеров?", ("Нет", "Да"), key="dendrogram_need_box")
        
        if dendrogram_need=="Да":
          if hierarchichal_df.shape[0]<=100:
            dendrogram_level = st.selectbox("Выберите уровень глубины дендрограммы",["Не выбрано"]+[i for i in range (3,hierarchichal_df.shape[0]+1)])
          else:
            dendrogram_level = st.selectbox("Выберите уровень глубины дендрограммы",["Не выбрано"]+[i for i in range (3,100)])
            
          if dendrogram_level!="Не выбрано":
            hierarchy_dendrogram(hierarchichal_df, int(dendrogram_level))
             
        if hierarchichal_df.shape[0]<=100:
          hierarchy_cluster_quan = st.selectbox("Укажите количество кластеров",["Не выбрано"]+[i for i in range(3, hierarchichal_df.shape[0]+1)], key="clusters_quan_hierarchy")
          # hierarchy_cluster_quan = st.selectbox("Укажите количество кластеров",["Не выбрано"]+[i for i in range (3,hierarchichal_df.shape[0]+1)], )
        else:
          hierarchy_cluster_quan = st.selectbox("Укажите количество кластеров",["Не выбрано"]+[i for i in range (3,100)], key="clusters_quan_hierarchy")
          
        def hierarchy_clusterisation(hierarchichal_df, quan_of_clusters):
          model = AgglomerativeClustering(quan_of_clusters)
          cluster_labels = model.fit_predict(hierarchichal_df)
          hierarchichal_df["Номер кластера"] = cluster_labels
          st.session_state["current_hierarchichal_df"] = hierarchichal_df
          return hierarchichal_df
  
        if hierarchy_cluster_quan!="Не выбрано": 
          hierarchichal_df = hierarchy_clusterisation(hierarchichal_df, hierarchy_cluster_quan)
          st.session_state["current_hierarchichal_df"]
          buffer = io.BytesIO()
          
          with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
              # Write each dataframe to a different worksheet.
              st.session_state["current_hierarchichal_df"].to_excel(writer, sheet_name='hierarchy')
          
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
    st.write('Загрузите файл и проведите предобработку загруженных данных во вкладке "Импорт и предобработка данных"')

with st.expander('Кластеризация методом DBSCAN'):  
  if unploaded_file:
    if df_state:
      if df.shape[0]>=3:
        dbscan_df = df.copy()
  
        eps_to_use = st.number_input("Выберите параметр эпсилон", value=0.01)
        min_samples_to_use = st.selectbox("Выберите параметр min_samples", [i for i in range(len(df)+1)])
        
        def dbscan_clusterisation():
            model = DBSCAN(eps=eps_to_use, min_samples=min_samples_to_use)
            cluster_labels = model.fit_predict(dbscan_df)
            dbscan_df["Номер кластера"] = cluster_labels
            st.session_state["current_dbscan_df"] = dbscan_df
            return dbscan_df

        if 'dbscan_button_clicked' not in st.session_state:
          st.session_state.dbscan_button_clicked = False
    
        def dbscan_button_on_click():
            st.session_state.dbscan_button_clicked = True
          
        dbscan_button = st.button("Применить кластеризацию с заданными параметрами", on_click=dbscan_button_on_click)

        if st.session_state.dbscan_button_clicked: 
          dbscan_df = dbscan_clusterisation()
          st.session_state["current_dbscan_df"]
          buffer = io.BytesIO()
          
          
          with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Write each dataframe to a different worksheet.
              if "current_dbscan_df" in st.session_state:
                  st.session_state["current_dbscan_df"].to_excel(writer, sheet_name='dbscan')
              
                  # Close the Pandas Excel writer and output the Excel file to the buffer
                  writer.close()
              
                  st.download_button(
                      label="Загрузить датафрейм в эксель-файл",
                      data=buffer,
                      file_name="dataframe_hierarchy_algorithm.xlsx",
                      mime="application/vnd.ms-excel",
                      key="dbscan_download"
                  )

      
      else:
        st.write("В датасете меньше трёх строк, кластеризация бессмысленна. Увеличьте количество строк или измените параметры подгтовки датасета, если в исходном датасете строк больше")
    else:
      st.write("Проведите предобработку загруженных данных")
  else:
    st.write('Загрузите файл и проведите предобработку загруженных данных во вкладке "Импорт и предобработка данных"')

  

    




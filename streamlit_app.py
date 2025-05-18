import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
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


st.title('💻 Кластеризация на основе файлов эксель')

st.info("Это веб-приложение для кластеризации ваших данных, хранящихся в эксель-файлах")

with st.expander('Импорт данных'):

  unploaded_file = st.file_uploader(label="Загрузите свой файл")

  if unploaded_file:
    # col_numbers = ["В датасете нет колонки для индекса"] + [i for i in range (1,df.shape[1]+1)]
    col_index_change = st.selectbox("Выберите вариант индексирования", ["В датасете нет колонки для индекса",
                                                                                         "Индексом датасета является первый столбец"])
    if col_index_change:
      if col_index_change == "В датасете нет колонки для индекса":
        df = pd.read_excel(unploaded_file)
        df
      else:
        df = pd.read_excel(unploaded_file, index_col = 0)
        df

      
    # df = pd.read_excel(unploaded_file)
    # # df
    # col_titles = df.columns.values.tolist()
    # df.set_index(col_titles[1])
    # # col_titles
    # df
    
with st.expander('Подготовка датасета'):
  if unploaded_file:
    st.header("Введите параметры подготовки данных")

    
    null_transform = st.selectbox("Выберите вариант обработки пустых значений переменных", ("Удалять строки, содержащие пустые значения", 
                                                                           "Заменять пустые значения на моду в колонке"
                                                                          ))

    categorial_to_numerical = st.selectbox("Выберите вариант преобразования категориальных переменных в численные", ("Удалять строки, содержащие пустые значения", 
                                                                           "Заменять пустые значения на среднее значение в колонке",
                                                                           "Заменять пустые значения на моду в колонке"
                                                                          ))

    
    if null_transform=="Удалять строки, содержащие пустые значения":
      df = df.dropna()
    else:
      for column in df.columns:
        df[column] = df[column].fillna(df[column].mode()[0])
  
    # col_index_categorical = st.selectbox("Выберите, вариант обработки пустых значений категориальных переменных", ("Удалять строки, содержащие пустые значения", 
    #                                                                        "Заменять пустые значения на моду в колонке"
    #                                                                       ))
    # preparation_button = st.button("Сохранить")
    # if preparation_button:
    #   st.write("Данные сохранены")

  else:
    st.write('Загрузите файл во вкладке "Импорт данных"')

# st.session_state["elbow_method_plot"] = None
# myplot = None


with st.expander('Кластеризация методом k-means++'):
      
  if unploaded_file:
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
          scaled_df = scaler.fit_transform(df)
          for quan_of_clusters in range(2, max_clusters_quan+1):
              model = KMeans(n_clusters=quan_of_clusters, init="k-means++")
              model.fit(scaled_df)
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
          scaled_df = scaler.fit_transform(df)
          model = KMeans(n_clusters = quan_of_clusters, init = "k-means++")
          cluster_labels = model.fit_predict(scaled_df)
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
    st.write('Загрузите файл во вкладке "Импорт данных"')
    
with st.expander('Иерархическая кластеризация'):  
  if unploaded_file:
    if df.shape[0]>=3:
      st.write("Я здеся!")

      def hierarchy_dendrogram(df, level=31):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
        linkage_matrix = hierarchy.linkage(scaled_df.values, method="ward")
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
      # else:
        # hierarchy_cluster_quan = st.selectbox("Укажите количество кластеров",["Не выбрано"]+[i for i in range (3,100)], key="clusters_quan_hierarchy")
        
      def hierarchy_clusterisation(df, quan_of_clusters):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
        model = AgglomerativeClustering(quan_of_clusters)
        cluster_labels = model.fit_predict(scaled_df)
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
    st.write('Загрузите файл во вкладке "Импорт данных"')

with st.expander('Метод DBSCAN'):  
  if unploaded_file:
    if df.shape[0]>=3:
      st.write("Я тууууут!")
      
      # dbscan = DBSCAN()

      epsilon_def_state = st.selectbox("Требуется ли помощь в определении параметра эпсилон?", ["Нет", "Да"])
      if epsilon_def_state == "Да":
        points = df.values
        hull = ConvexHull(points)
        hullpoints = points[hull.vertices,:]
        longest_dist = cdist(hullpoints, hullpoints, metric='euclidean').max()
   
        def euclidean_distance(p1: List[float], p2: List[float]) -> float:
          return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
  
        def closest_pair_recursive(points: List[List[float]], depth: int = 0) -> float:
            n = len(points)
            if n <= 3:
                # Brute force for small number of points
                return min(
                    euclidean_distance(points[i], points[j])
                    for i in range(n)
                    for j in range(i + 1, n)
                )
        
            # Select axis based on depth (cycle through dimensions)
            k = len(points[0])  # Dimension
            axis = depth % k
        
            # Sort points along current axis and split
            points.sort(key=lambda x: x[axis])
            mid = n // 2
            left = points[:mid]
            right = points[mid:]
        
            # Recurse on both halves
            d_left = closest_pair_recursive(left, depth + 1)
            d_right = closest_pair_recursive(right, depth + 1)
            d = min(d_left, d_right)
        
            # Build strip near the splitting plane
            mid_value = points[mid][axis]
            strip = [p for p in points if abs(p[axis] - mid_value) < d]
        
            # Compare points in the strip across the splitting line
            min_d_strip = d
            for i in range(len(strip)):
                for j in range(i + 1, len(strip)):
                    if euclidean_distance(strip[i], strip[j]) < min_d_strip:
                        min_d_strip = euclidean_distance(strip[i], strip[j])
        
            return min(d, min_d_strip)
        
        def closest_pair(points: List[List[float]]) -> float:
            if len(points) < 2:
                return float('inf')
            return closest_pair_recursive(points)
          
        shortest_dist = closest_pair(points.tolist())
   

        outlier_percent = []
        number_of_outliers = []
        quan_of_clusters_eps_list = []

        for eps in np.linspace(shortest_dist, longest_dist, len(points)):
            dbscan = DBSCAN(eps=eps)
            dbscan.fit(df)
            number_of_outliers.append(np.sum(dbscan.labels_ == -1))
            percent_outliers = 100*np.sum(dbscan.labels_ == -1) / len(points)
            outlier_percent.append(percent_outliers)
            quan_of_clusters_eps = len(np.unique(dbscan.labels_))
            quan_of_clusters_eps_list.append(quan_of_clusters_eps)
          
        fig, ax = plt.subplots()
        sns.lineplot(x=np.linspace(shortest_dist, longest_dist, len(points)), y=number_of_outliers)
        # ax.set_title("Количество выбросов", fontsize=10)
        # st.pyplot(fig)
        
        sns.lineplot(x=np.linspace(shortest_dist, longest_dist, len(points)), y=outlier_percent)
        # ax.set_title("Процент выбросов", fontsize=30)
        sns.lineplot(x=np.linspace(shortest_dist, longest_dist, len(points)), y=quan_of_clusters_eps_list)
        st.pyplot(fig)
        
          

        


    
    else:
      st.write("В датасете меньше трёх строк, кластеризация бессмысленна. Увеличьте количество строк или измените параметры подгтовки датасета, если в исходном датасете строк больше")
  else:
    st.write('Загрузите файл во вкладке "Импорт данных"')

  

    




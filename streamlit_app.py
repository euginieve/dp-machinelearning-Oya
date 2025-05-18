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
        
        # Plot dendrogram on the provided axis
        dendrogram(linkage_matrix, truncate_mode="level", p=level-1, ax=ax)
        
        # Display the figure in Streamlit
        st.pyplot(fig)
        # plt.figure(figsize=(20,10), dpi=200)
        # plt.title(label="Дендрограмма", fontsize=30)
        # dendro = dendrogram(linkage_matrix, truncate_mode="level", p=level-1)
        # st.pyplot()

        # fig, ax = plt.subplots(figsize=(20, 10), dpi=200)
        # ax.set_title("Дендрограмма", fontsize=30)
        
        # Plot dendrogram on the provided axis
        # dendrogram(linkage_matrix, truncate_mode="level", p=level-1, ax=ax)
        # st.write([y[1] for y in dendrogram('dcoord')])
        
        # return [y[1] for y in dendrogram['dcoord']]
        
        
        # Display the figure in Streamlit
        # st.pyplot(fig)
        
        # plt.figure(figsize=(20,10), dpi=200)
        # plt.title(label="Дендрограмма", fontsize=30)
        # dendro = dendrogram(linkage_matrix, truncate_mode="level", p=level-1)
        # fig, ax = plt.subplots()
        # st.pyplot(fig)
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
      st.write("Я тута!")
      
      # dbscan = DBSCAN()

      epsilon_def_state = st.selectbox("Требуется ли помощь в определении параметра эпсилон?", ["Нет", "Да"])
      if epsilon_def_state == "Да":
        points = df.values
        hull = ConvexHull(points)
        hullpoints = points[hull.vertices,:]
        longest_dist = cdist(hullpoints, hullpoints, metric='euclidean').max()

        def distance(p1, p2):
          return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to find the minimum distance in the strip
        def stripClosest(strip, d):
            min_dist = d
        
            # Sort points in the strip by their y-coordinate
            strip.sort(key=lambda point: point[1])
        
            # Compare each point in the strip
            for i in range(len(strip)):
                for j in range(i + 1, len(strip)):
                    if (strip[j][1] - strip[i][1]) < min_dist:
                        min_dist = min(min_dist, distance(strip[i], strip[j]))
                    else:
                        break
        
            return min_dist
        
        # Divide and conquer function to find the minimum distance
        def minDistUtil(points, left, right):
            
            # Base case brute force for 2 or fewer points
            if right - left <= 2:
                min_dist = float('inf')
                for i in range(left, right):
                    for j in range(i + 1, right):
                        min_dist = min(min_dist, distance(points[i], points[j]))
                return min_dist
        
            # Find the midpoint
            mid = (left + right) // 2
            mid_x = points[mid][0]
        
            # Recursively find the minimum distances
            # in the left and right halves
            dl = minDistUtil(points, left, mid)
            dr = minDistUtil(points, mid, right)
        
            d = min(dl, dr)
        
            # Build the strip of points within distance d from the midl
            strip = []
            for i in range(left, right):
                if abs(points[i][0] - mid_x) < d:
                    strip.append(points[i])
        
            # Find the minimum distance in the strip
            stripDist = stripClosest(strip, d)
        
            return min(d, stripDist)
        
        # Function to find the closest pair of points
        def minDistance(points):
            n = len(points)
        
            # Sort points by x-coordinate
            points.sort(key=lambda point: point[0])
        
            return minDistUtil(points, 0, n)

        st.write("lalalala")
    
    else:
      st.write("В датасете меньше трёх строк, кластеризация бессмысленна. Увеличьте количество строк или измените параметры подгтовки датасета, если в исходном датасете строк больше")
  else:
    st.write('Загрузите файл во вкладке "Импорт данных"')

  

    




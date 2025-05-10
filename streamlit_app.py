import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title('💻 Кластеризация данных на основе таблиц в эксель-файлах')

st.info("Это веб-приложение для кластеризации ваших данных, хранящихся в эксель-файлах")

with st.expander('Данные для загрузки'):

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

    col_index_numerical = st.selectbox("Выберите, вариант обработки пустых значений численных переменных", ("Удалять строки, содержащие пустые значения", 
                                                                           "Заменять пустые значения на среднее значение в колонке",
                                                                           "Заменять пустые значения на моду в колонке"
                                                                          ))
    col_index_categorical = st.selectbox("Выберите, вариант обработки пустых значений категориальных переменных", ("Удалять строки, содержащие пустые значения", 
                                                                           "Заменять пустые значения на моду в колонке"
                                                                          ))
    # preparation_button = st.button("Сохранить")
    # if preparation_button:
    #   st.write("Данные сохранены")

  else:
    st.write('Загрузите файл во вкладке "Данные для загрузки"')

with st.expander('Кластеризация методом k-means++'):
  elbow_method_need = st.selectbox("Требуется ли построить график локтя для лучшего проедставления о необходимом количестве кластеров?", ("Да", "Нет"))
  if elbow_method_need == "Да":
    def elbow_method(df, max_clusters_quan):
    
      try:
          if max_clusters_quan > len(df):
              print("Указанное максимальное количество кластеров превышает длину таблицы. Укажите меньшее количество.")
              return None
          if max_clusters_quan < 2:
              print("Указанное максимальное число кластеров меньше двух и, следовательно, не имеет смысла. Уквжите число кластеров юольшее или равное двум.")
              return None
          if max_clusters_quan != int(max_clusters_quan):
              print("Максимальное количество кластеров должно быть целым числом.")
              return None
              
          ssd = []
          scaler = StandardScaler()
          scaled_df = scaler.fit_transform(df)
          for quan_of_clusters in range(2, max_clusters_quan+1):
              model = KMeans(n_clusters = quan_of_clusters, init = "k-means++")
              model.fit(scaled_df)
              ssd.append(model.inertia_)
          
          plt.plot(range(2, max_clusters_quan+1), ssd, "o--")
          plt.title("График локтя")
          return None
      
      except Exception as e:
          pst.write(f"Ошибка при использовании метода: {e}")
          return None
      
  if unploaded_file:   
      k_means_cluster_quan = st.text_input("Введите количество кластеров")
      if k_means_cluster_quan and not k_means_cluster_quan.isdigit():
        
        st.write("Количество должно быть числом")
        

      def k_means_plus_plus(df, quan_of_clusters):
        try:
          scaler = StandardScaler()
          scaled_df = scaler.fit_transform(df)
          model = KMeans(n_clusters = quan_of_clusters, init = "k-means++")
          cluster_labels = model.fit_predict(scaled_df)
          df["Номер кластера"] = cluster_labels
          return df
                         
        except Exception as e:
          st.write(f"Ошибка при кластеризации {e}")
          return None
      if k_means_cluster_quan and k_means_cluster_quan.isdigit(): 
        df = k_means_plus_plus(df, int(k_means_cluster_quan))
        df
  else:
    st.write('Загрузите файл во вкладке "Данные для загрузки"')
    
    



import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title('🖥Кластеризация данных на основе таблиц в эксель-файлах')

st.info("Это веб-приложение для кластеризации ваших данных, хранящихся в эксель-файлах")

with st.expander('Данные для загрузки'):

  unploaded_file = st.file_uploader(label="Загрузите свой файл")

  if unploaded_file:
    df = pd.read_excel(unploaded_file, index_col=0)
    df
    
with st.expander('Подготовка датасета'):
  if unploaded_file:
    st.header("Введите параметры подготовки данных")
    col_numbers = ["В датасете нет колонки для индекса"] + [i for i in range (1,df.shape[1]+1)]
    col_index = st.selectbox("Выберите номер колонки, которую желаете сделать индексом", col_numbers)
    # if col_index == "В датасете нет колонки для индекса":
    #   col_index = 0
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
        k_means_plus_plus(df, int(k_means_cluster_quan))
  else:
    st.write('Загрузите файл во вкладке "Данные для загрузки"')
    



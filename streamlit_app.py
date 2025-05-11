import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import io

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
      
  if unploaded_file:
    if df.shape[0]>=3:
      elbow_method_need = st.selectbox("Требуется ли построить график локтя для лучшего представления о необходимом количестве кластеров?", ("Нет", "Да"))
      if elbow_method_need == "Да":
        
        if df.shape[0]<=100:
          clusters_quan_elbow_method = st.selectbox("Укажите максимальное количество кластеров",[i for i in range (3,df.shape[0]+1)])
        else:
          clusters_quan_elbow_method = st.selectbox("Укажите максимальное количество кластеров",[i for i in range (3,100)])
    
        def elbow_method(df, max_clusters_quan):    
          # st.session_state.clicked = True
          ssd = []
          scaler = StandardScaler()
          scaled_df = scaler.fit_transform(df)
          for quan_of_clusters in range(2, max_clusters_quan+1):
              model = KMeans(n_clusters = quan_of_clusters, init = "k-means++")
              model.fit(scaled_df)
              ssd.append(model.inertia_)
          plt.plot(range(2, max_clusters_quan+1), ssd, "o--")
          plt.title("График локтя")
          # Get the current axes
          ax = plt.gca()
          # Set x-axis to only display integers
          ax.xaxis.set_major_locator(MaxNLocator(integer=True))
          # st.session_state["elbow_plot"] = st.pyplot(plt)
          return st.pyplot(plt)

        # st.session_state["elbow_method_button_clicked"] 
        def click_button():
          st.session_state["elbow_method_button_clicked"] = True
          # elbow_method(df, clusters_quan_elbow_method)
        
        
        # elbow_method_button = st.button("Построить график локтя")
  
        
        # if st.session_state["elbow_method_button_clicked"]:
        # if elbow_method_button:
        st.session_state["elbow_method_plot"] = elbow_method(df,clusters_quan_elbow_method)
        st.session_state["elbow_method_plot"]
          # st.session_state["elbow_plot"]
    
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
          st.session_state["current_df"] = df
          return df
                         
        except Exception as e:
          st.write(f"Ошибка при кластеризации {e}")
          return None
          
      if k_means_cluster_quan and k_means_cluster_quan.isdigit(): 
        df = k_means_plus_plus(df, int(k_means_cluster_quan))
        st.session_state["current_df"]
      # Create a Pandas Excel writer using XlsxWriter as the engine.
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Write each dataframe to a different worksheet.
            st.session_state["current_df"].to_excel(writer, sheet_name='k_means')
        
            # Close the Pandas Excel writer and output the Excel file to the buffer
            writer.close()
        
            st.download_button(
                label="Download Excel worksheets",
                data=buffer,
                file_name="dataframe_k_means.xlsx",
                mime="application/vnd.ms-excel"
            )
            
    else:
      st.write("В датасете меньше трёх строк, кластеризация бессмысленна. Увеличьте количество строк или измените параметры подгтовки датасета, если в исходном датасете строк больше")
  else:
    st.write('Загрузите файл во вкладке "Данные для загрузки"')
    
    



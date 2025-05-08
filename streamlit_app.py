import streamlit as st
import pandas as pd

st.title('🖥Кластеризация данных на основе таблиц в эксель-файлах')

st.info("Это веб-приложение для кластеризации ваших данных, хранящихся в эксель-файлах")

with st.expander('Данные для загрузки'):

  unploaded_file = st.file_uploader(label="Загрузите свой файл")

  if unploaded_file:
    df = pd.read_excel(unploaded_file)
    df
    
with st.expander('Подготовка датасета'):
  if unploaded_file:
    st.header("Введите параметры подготовки данных")
    col_numbers = ["В датасете нет колонки для индекса"] + [i for i in range (1,df.shape[1]+1)]
    col_index = st.selectbox("Выберите номер колонки, которую желаете сделать индексом", col_numbers)
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
      if col_numbers and col_index_numerical and col_index_categorical:
        st.write('Зыгрузите')
      else:
        st.write('Выберите параметры подготовки датасета во вкладке "Подготовка датасета"')
  else:
    st.write('Загрузите файл во вкладке "Данные для загрузки"')




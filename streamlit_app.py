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
    
  else:
    st.write('Зыгрузите файл во вкладке "Данные для загрузки"')




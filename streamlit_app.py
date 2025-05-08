import streamlit as st
import pandas as pd

st.title('🖥Кластеризация данных на основе таблиц в эксель-файлах')

st.info("Это веб-приложение для кластеризации ваших данных, хранящихся в эксель-файлах")

df = st.file_uploader(label="Загрузите свой файл")




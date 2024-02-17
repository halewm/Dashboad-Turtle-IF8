import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load datasets
hour_df = pd.read_csv("hour.csv")
day_df = pd.read_csv("day.csv")

# Filter data for years 2011 and 2012
hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])
hour_df_2011 = hour_df[hour_df['dteday'].dt.year == 2011]
hour_df_2012 = hour_df[hour_df['dteday'].dt.year == 2012]

# Function to perform classification
def perform_classification(df):
    X = df[['temp', 'hum', 'windspeed', 'workingday', 'hr']]
    y = df['cnt'] > df['cnt'].mean()  # Binary classification based on average count
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Function to perform K-Means clustering
def perform_clustering(df, num_clusters):
    X = df[['temp', 'hum', 'windspeed', 'cnt']]
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    df['cluster'] = kmeans.labels_
    return df

# Function to plot histogram or KDE plot for each day in each month
def plot_daily_distribution_by_month(df):
    df['month'] = df['dteday'].dt.month
    df['day'] = df['dteday'].dt.day

    # Group by month and day, then calculate the mean count for each day
    daily_data = df.groupby(['month', 'day'])['cnt'].mean().reset_index()

    # Plot
    fig = px.line(daily_data, x='day', y='cnt', color='month', title="Distribusi Jumlah peminjaman Sepeda per Hari dalam Setiap Bulan")
    return fig

# Function to find peak hours
def find_peak_hours(df):
    peak_hours = df.groupby('hr')['cnt'].mean().sort_values(ascending=False).head(2).index
    return peak_hours

# Function to plot line chart for monthly bike usage
def plot_monthly_trend(df, year):
    df['month'] = df['dteday'].dt.strftime('%B')
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    monthly_data = df.groupby('month')['cnt'].sum().reindex(month_order).reset_index()
    fig = px.line(monthly_data, x='month', y='cnt', title=f"Tren peminjaman Sepeda Setiap Bulannya pada Tahun {year}")
    return fig

# Function to plot bar chart for factors affecting bike usage
def plot_factors(df, year):
    fig = px.box(df, x='mnth', y='cnt', title=f"Faktor yang Mempengaruhi peminjaman Sepeda Setiap Bulannya pada Tahun {year}", 
                 labels={'mnth': 'Bulan', 'cnt': 'Jumlah Pengguna'})
    return fig

# Function to plot line chart for daily bike usage distribution
def plot_daily_trend(df, year):
    df['day'] = pd.to_datetime(df['dteday']).dt.strftime('%Y-%m-%d')
    daily_data = df.groupby('day')['cnt'].sum().reset_index()
    daily_data = daily_data[daily_data['day'].str.startswith(str(year))]  # Filter data by year
    fig = px.line(daily_data, x='day', y='cnt', title=f"Tren peminjaman Sepeda Setiap Harinya pada Tahun {year}")
    return fig

# Function to plot line chart for hourly bike usage distribution
def plot_hourly_distribution(df, year):
    hourly_data = df.groupby('hr')['cnt'].mean().reset_index()
    fig = px.line(hourly_data, x='hr', y='cnt', title=f"Distribusi Jumlah peminjaman Sepeda per Jam dalam Sehari pada Tahun {year}")
    return fig

# Analisis Korelasi
st.header("Analisis peminjaman Sepeda (Bike Sharing)")

# Sidebar with options menu
with st.sidebar:
    selected = option_menu('Menu', ['Dashboard'], icons=["easel2", "graph-up"], menu_icon="cast", default_index=0)

# Page 1: Dashboard
if selected == 'Dashboard':
    st.header("Data pada Tahun 2011 dan 2012")

    st.plotly_chart(plot_monthly_trend(hour_df_2011, 2011))
    with st.expander("Penjelasan Tren peminjaman Sepeda Setiap Bulannya"):
        st.write("Dilihat dari grafik tersebut, Tren peminjaman sepeda mengalami peningkatan dari bulan Januari - Juni Lalu, mengalami penurunan hingga bulan Desember.")
    st.plotly_chart(plot_monthly_trend(hour_df_2012, 2012))
    with st.expander("Penjelasan Tren peminjaman Sepeda Setiap Bulannya"):
        st.write("Dilihat dari grafik tersebut,  Tren penggunaa Sepeda mengalami kenaikan dari bulan Sanuari - September dan mengalami penurunan hingga Desember.")

    st.plotly_chart(plot_factors(hour_df_2011, 2011))
    with st.expander("Penjelasan Faktor peminjaman Sepeda Setiap Bulannya di 2011"):
        st.write("Dilihat dari grafik tersebut, terlihat bahwa pengguna peminjaman sepeda menaik dari bulan 5 - 9 yang umumnya adalah musim panas atau sedikit hujan.")
    st.plotly_chart(plot_factors(hour_df_2012, 2012))
    with st.expander("Penjelasan Faktor peminjaman Sepeda Setiap Bulannya di 2012"):
        st.write("Dilihat dari grafik tersebut, terlihat bahwa pengguna peminjaman sepeda cukup stabil, dikarenakan angin yang cukup kencang pada tahun tersebut.")
        
    st.plotly_chart(plot_daily_trend(day_df, 2011))
    with st.expander("Tren Peminjman sepeda perhari selama 2011 "):
        st.write("Dilihat dari grafik tersebut,Tren Peminjman sepeda menaik pada hari - hari libur, khususnya pada bulan Mei - Juli dikarenakan bulan tersebut adalah hari libur.")
        
    st.plotly_chart(plot_daily_trend(day_df, 2012))
    with st.expander("Penjelasan Distribusi Jumlah peminjaman Sepeda per Jam dalam Sehari"):
        st.write("Dilihat dari grafik tersebut, Tren Peminjman sepeda menaik pada hari - hari libur, khususnya pada bulan Mei - November dikarenakan bulan tersebut banyak hari libur.")
        
    st.plotly_chart(plot_hourly_distribution(hour_df_2011, 2011))
    with st.expander("Penjelasan Distribusi jumlah rata-rata penyewa sepeda perjam dalam sehari pada tahun 2011"):
        st.write("Dilihat dari grafik tersebut, Distribusi jumlah rata-rata penyewa sepeda perjam menaik dari jam 8 - 17 dikarena kan kelembapan berkurang pada jam tersebut")
        
    st.plotly_chart(plot_hourly_distribution(hour_df_2012, 2012))
    with st.expander("Penjelasan Distribusi jumlah rata-rata penyewa sepeda perjam dalam sehari pada tahun 2012"):
        st.write("Sama halnya dengan tahun 2011, Distribusi jumlah rata-rata penyewa sepeda perjam menaik dari jam 8 - 17 dikarena kan kelembapan berkurang pada jam tersebut.")
        
    
    st.plotly_chart(plot_daily_distribution_by_month(hour_df))   
    with st.expander("Penjelasan Distribusi peminjaman sepeda perhari dalam setiap bulan selama setahun"):
        st.write("dari grafik tersebut, bulan dengan rata-rata peminjaman tertinggi adalah bulan september khususnya ditanggal 25, dan yang terendah adalah bulan januari khususnya tanggal 5 ")     
     
    peak_hours_2011 = find_peak_hours(hour_df_2011)
    st.write("Jam peminjaman tertinggi:", peak_hours_2011)
    with st.expander("Penjelasan Jam peminjaman tertinggi"):
        st.write("Jam peminjaman tertinggi adalah 18 & 17 jam")     
    
    num_clusters = st.slider("Jumlah cluster:", min_value=2, max_value=10, value=3)
    clustered_df = perform_clustering(hour_df, num_clusters)
    st.write("Data dengan Penambahan Label Cluster:", clustered_df.head())
    with st.expander("Penjelasan Data dengan Penambahan Label Cluster"):
        st.write("kita bisa melihat apa yang terjadi pada cluster apabila kita menaikkanya")
    
    classification_accuracy = perform_classification(hour_df)
    st.write("Akurasi Klasifikasi:", classification_accuracy)
    with st.expander("Penjelasan Akurasi Klasifikasi"):
        st.write("Akurasi dari data data ini adalah 0.8811852704257768")

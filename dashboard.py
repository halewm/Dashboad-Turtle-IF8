import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# Memuat dataset
day_df = pd.read_csv('day.csv')
hour_df = pd.read_csv('hour.csv')

# Fungsi untuk mengonversi nomor bulan menjadi nama bulan
def get_month_name(month_number):
    months = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
    return months[month_number - 1]

# Fungsi untuk membuat diagram batang dengan label legenda
def barplot_with_legend(x, y, hue, data, title, xlabel, ylabel, legend_labels):
    # Membuat objek gambar dan sumbu
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax)
    ax.legend( labels=[legend_labels[i] for i in sorted(legend_labels)])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Menampilkan gambar menggunakan st.pyplot()
    st.pyplot(fig)

# Fungsi untuk membuat heatmap
def plot_heatmap(data, title):
    # Membuat objek gambar dan sumbu
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title(title)
    # Menampilkan gambar menggunakan st.pyplot()
    st.pyplot(fig)

# Mendefinisikan fungsi untuk melakukan analisis clustering
def clustering_analysis(hour_df):
    # Mendefinisikan jumlah klaster
    n_clusters = 5

    # Membuat objek K-Means dan melatihnya pada fitur yang relevan
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(hour_df[['temp', 'atemp', 'hum', 'windspeed']])  # Menggunakan fitur yang tersedia di dalam dataset

    # Memberikan label klaster pada hour_df
    hour_df['cluster'] = kmeans.labels_

    # Menampilkan pusat klaster
    st.write("Pusat Klaster:")
    st.write(kmeans.cluster_centers_)

    # Menampilkan jumlah data dalam setiap klaster
    st.write("Jumlah data dalam setiap klaster:")
    st.write(hour_df['cluster'].value_counts())

    # Plot klaster pada scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='temp', y='hum', hue='cluster', data=hour_df, palette='tab10', ax=ax)
    plt.title('Analisis Klaster K-Means')
    plt.xlabel('Suhu')
    plt.ylabel('Kelembaban')
    st.pyplot(fig)

# Sidebar
st.sidebar.title('Pilihan Analisis Data')
analysis_choice = st.sidebar.radio("Pilih Analisis", ('Analisis Bulanan 2011', 'Analisis Bulanan 2012', 'Analisis Hari Libur 2011', 'Analisis Hari Libur 2012', 'Heatmap Korelasi', 'Analisis Klaster'))
# Konten Utama
st.title('Analisis Data Peminjaman Sepeda')

if analysis_choice == 'Analisis Bulanan 2011':
    st.header('Analisis Bulanan 2011')
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    day_df_2011 = day_df[day_df['dteday'].dt.year == 2011]
    month_df_2011 = day_df_2011['mnth'].unique()
    rataRata_2011 = day_df_2011.groupby('mnth')['cnt'].mean()
    rataRata_cuaca_2011 = day_df_2011.groupby('mnth')['temp'].mean()
    rataRata_hum_2011 = day_df_2011.groupby('mnth')['hum'].mean()
    legend_labels = {
        1: 'Cerah, Sedikit awan, Sebagian berawan, Sebagian berawan',
        2: 'Kabut + Berawan, Kabut + Awan pecah, Kabut + Sedikit awan, Kabut',
        3: 'Hujan Ringan, Hujan Ringan + Petir + Awan terpencar, Hujan Ringan + Awan terpencar',
        4: 'Hujan Lebat + Pecahan Es + Petir + Kabut, Salju + Kabut'
    }
    barplot_with_legend('mnth', 'cnt', 'weathersit', day_df_2011, 'Jumlah total sepeda berdasarkan situasi cuaca setiap bulannya (2011)', 'Bulan', 'Total', legend_labels)

elif analysis_choice == 'Analisis Bulanan 2012':
    st.header('Analisis Bulanan 2012')
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    day_df_2012 = day_df[day_df['dteday'].dt.year == 2012]
    month_df_2012 = day_df_2012['mnth'].unique()
    rataRata_2012 = day_df_2012.groupby('mnth')['cnt'].mean()
    barplot_with_legend('mnth', 'cnt', 'workingday', day_df_2012, 'Jumlah total sepeda berdasarkan hari kerja setiap bulan (2012)', 'Bulan', 'Total', {})

elif analysis_choice == 'Analisis Hari Libur 2011':
    st.header('Analisis Hari Libur 2011')
    # Filter data untuk hari libur pada tahun 2011
    holiday_data_2011 = day_df[(day_df['holiday'] == 1) & (day_df['yr'] == 0)]
    
    # Plot pengaruh cuaca terhadap penggunaan sepeda selama hari libur tahun 2011
    fig, ax = plt.subplots()
    sns.scatterplot(x='weathersit', y='cnt', data=holiday_data_2011, ax=ax)
    ax.set_title('Pengaruh Cuaca terhadap Penggunaan Sepeda selama Hari Libur (2011)')
    ax.set_xlabel('Situasi Cuaca')
    ax.set_ylabel('Jumlah Sepeda yang Dipinjam')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Cerah', 'Kabut + Berawan', 'Salju Ringan/Hujan Ringan'])
    
    # Menampilkan plot menggunakan st.pyplot()
    st.pyplot(fig)

elif analysis_choice == 'Analisis Hari Libur 2012':
    st.header('Analisis Hari Libur 2012')
    # Filter data untuk hari libur pada tahun 2012
    holiday_data_2012 = day_df[(day_df['holiday'] == 1) & (day_df['yr'] == 1)]
    
    # Plot pengaruh cuaca terhadap penggunaan sepeda selama hari libur tahun 2012
    fig, ax = plt.subplots()
    sns.scatterplot(x='weathersit', y='cnt', data=holiday_data_2012, ax=ax)
    ax.set_title('Pengaruh Cuaca terhadap Penggunaan Sepeda selama Hari Libur (2012)')
    ax.set_xlabel('Situasi Cuaca')
    ax.set_ylabel('Jumlah Sepeda yang Dipinjam')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Cerah', 'Kabut + Berawan', 'Salju Ringan/Hujan Ringan'])
    
    # Menampilkan plot menggunakan st.pyplot()
    st.pyplot(fig)

elif analysis_choice == 'Heatmap Korelasi':
    st.header('Heatmap Korelasi')
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    day_df_2011 = day_df[day_df['dteday'].dt.year == 2011]
    month_df_2011 = day_df_2011['mnth'].unique()
    rataRata_2011 = day_df_2011.groupby('mnth')['cnt'].mean()
    rataRata_cuaca_2011 = day_df_2011.groupby('mnth')['temp'].mean()
    rataRata_hum_2011 = day_df_2011.groupby('mnth')['hum'].mean()
    merged_df = pd.merge(rataRata_2011, rataRata_cuaca_2011, on='mnth', suffixes=('_cnt', '_temp'))
    merged_df_2011 = pd.merge(merged_df, rataRata_hum_2011, on='mnth')
    correlation_matrix_2011 = merged_df_2011.corr()
    plot_heatmap(correlation_matrix_2011, 'Heatmap Korelasi: Suhu, Kelembaban, dan Penggunaan Sepeda')

elif analysis_choice == 'Analisis Klaster':
    st.header('Analisis Klaster')
    clustering_analysis(hour_df)

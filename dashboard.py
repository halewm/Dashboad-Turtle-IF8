import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
day_df = pd.read_csv('day.csv')
hour_df = pd.read_csv('hour.csv')

# Function to convert month number to month name
def get_month_name(month_number):
    months = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
    return months[month_number - 1]

# Function to plot bar chart with legend labels
def barplot_with_legend(x, y, hue, data, title, xlabel, ylabel, legend_labels):
    # Create a figure object and store it in fig
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax)
    ax.legend(title='Weather Situation', loc='upper right', bbox_to_anchor=(1.25, 1), labels=[legend_labels[i] for i in sorted(legend_labels)])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Pass fig as an argument to st.pyplot()
    st.pyplot(fig)

# Function to plot heatmap
def plot_heatmap(data, title):
    # Create a figure object and store it in fig
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title(title)
    # Pass fig as an argument to st.pyplot()
    st.pyplot(fig)

# Sidebar
st.sidebar.title('Data Analysis Options')
analysis_choice = st.sidebar.radio("Choose Analysis", ('2011 Monthly Analysis', '2012 Monthly Analysis', 'Correlation Heatmap'))

# Main content
st.title('Bicycle Sharing Data Analysis')

if analysis_choice == '2011 Monthly Analysis':
    st.header('2011 Monthly Analysis')
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    day_df_2011 = day_df[day_df['dteday'].dt.year == 2011]
    month_df_2011 = day_df_2011['mnth'].unique()
    rataRata_2011 = day_df_2011.groupby('mnth')['cnt'].mean()
    rataRata_cuaca_2011 = day_df_2011.groupby('mnth')['temp'].mean()
    rataRata_hum_2011 = day_df_2011.groupby('mnth')['hum'].mean()
    legend_labels = {
        1: 'Clear, Few clouds, Partly cloudy, Partly cloudy',
        2: 'Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist',
        3: 'Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds',
        4: 'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog'
    }
    barplot_with_legend('mnth', 'cnt', 'weathersit', day_df_2011, 'Jumlah total sepeda berdasarkan situasi cuaca (weather situation) setiap bulannya (2011)', 'Month', 'Total Count', legend_labels)

elif analysis_choice == '2012 Monthly Analysis':
    st.header('2012 Monthly Analysis')
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    day_df_2012 = day_df[day_df['dteday'].dt.year == 2012]
    month_df_2012 = day_df_2012['mnth'].unique()
    rataRata_2012 = day_df_2012.groupby('mnth')['cnt'].mean()
    barplot_with_legend('mnth', 'cnt', 'workingday', day_df_2012, 'Jumlah total sepeda berdasarkan hari kerja setiap bulan (2012)', 'Month', 'Total Count', {'0': 'No', '1': 'Yes'})

elif analysis_choice == 'Correlation Heatmap':
    st.header('Correlation Heatmap')
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    day_df_2011 = day_df[day_df['dteday'].dt.year == 2011]
    month_df_2011 = day_df_2011['mnth'].unique()
    rataRata_2011 = day_df_2011.groupby('mnth')['cnt'].mean()
    rataRata_cuaca_2011 = day_df_2011.groupby('mnth')['temp'].mean()
    rataRata_hum_2011 = day_df_2011.groupby('mnth')['hum'].mean()
    merged_df = pd.merge(rataRata_2011, rataRata_cuaca_2011, on='mnth', suffixes=('_cnt', '_temp'))
    merged_df_2011 = pd.merge(merged_df, rataRata_hum_2011, on='mnth')
    correlation_matrix_2011 = merged_df_2011.corr()
    plot_heatmap(correlation_matrix_2011, 'Correlation Heatmap: Suhu, Kelembaban, dan Penggunaan Sepeda')

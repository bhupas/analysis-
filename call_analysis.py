import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import os
from datetime import datetime, timedelta
import numpy as np

# Streamlit configuration
st.set_page_config(page_title="📞 Enhanced Call Analysis Report", layout="wide")

# Title and Description
st.title("📞 Enhanced Call Analysis Report")
st.write("""
This comprehensive report provides an **in-depth analysis** of call patterns for bhruna.
""")

# Load and process data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['call.ended_timestamp'] = pd.to_datetime(data['call.ended_timestamp'])
    data['call_duration'] = (data['call.ended_timestamp'] - data['timestamp']).dt.total_seconds()
    
    # Additional time features
    data['month'] = data['timestamp'].dt.to_period('M').astype(str)
    data['day_of_week'] = data['timestamp'].dt.day_name()
    data['hour'] = data['timestamp'].dt.hour
    data['week'] = data['timestamp'].dt.isocalendar().week
    data['day'] = data['timestamp'].dt.date
    data['is_weekend'] = data['timestamp'].dt.weekday.isin([5, 6])
    data['time_of_day'] = pd.cut(data['hour'], 
                                bins=[0, 6, 12, 18, 24], 
                                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                right=False, include_lowest=True)
    
    return data

file_path = './_kafkasdaughter_page_1.csv'
if not os.path.exists(file_path):
    st.error(f"Data file not found at path: {file_path}")
else:
    data = load_data(file_path)
    user_name = 'method1909'
    user_call_data = data[data['userName'] == user_name].copy()

    if user_call_data.empty:
        st.warning(f"No call data found for user: {user_name}")
    else:
        # Calculate advanced metrics

        # Basic Statistics
        total_calls = len(user_call_data)
        total_duration_seconds = user_call_data['call_duration'].sum()
        total_duration_readable = str(timedelta(seconds=total_duration_seconds))
        avg_duration_seconds = user_call_data['call_duration'].mean()
        avg_duration_readable = str(timedelta(seconds=avg_duration_seconds))
        
        # Temporal Patterns
        calls_by_dow = user_call_data['day_of_week'].value_counts().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        )
        busiest_day = calls_by_dow.idxmax()
        quietest_day = calls_by_dow.idxmin()
        
        calls_by_tod = user_call_data['time_of_day'].value_counts().reindex(['Night', 'Morning', 'Afternoon', 'Evening'])
        busiest_time = calls_by_tod.idxmax()
        
        weekend_calls = user_call_data[user_call_data['is_weekend']].shape[0]
        weekday_calls = total_calls - weekend_calls
        weekend_percentage = (weekend_calls / total_calls) * 100
        
        # Duration Analysis
        duration_stats = user_call_data['call_duration'].describe()
        shortest_call = str(timedelta(seconds=duration_stats['min']))
        longest_call = str(timedelta(seconds=duration_stats['max']))
        median_duration = str(timedelta(seconds=duration_stats['50%']))
        std_deviation = str(timedelta(seconds=duration_stats['std']))
        
        # Trend Analysis
        monthly_calls = user_call_data.groupby('month').size().sort_index()
        call_trend = monthly_calls.pct_change().mean() * 100
        peak_month = monthly_calls.idxmax()
        lowest_month = monthly_calls.idxmin()
        
        # Cluster Analysis
        user_call_data = user_call_data.sort_values('timestamp')  # Ensure data is sorted by timestamp
        user_call_data['prev_call_gap'] = user_call_data['timestamp'].diff().dt.total_seconds()
        
        # Identify call clusters (calls within 1 hour of each other)
        cluster_threshold = 3600  # 1 hour in seconds
        user_call_data['cluster_id'] = (user_call_data['prev_call_gap'] > cluster_threshold).cumsum()
        
        clusters = user_call_data.groupby('cluster_id').agg({
            'timestamp': ['count', 'min', 'max'],
            'call_duration': 'sum'
        })
        
        clusters.columns = ['calls_in_cluster', 'cluster_start', 'cluster_end', 'total_duration']
        clusters['cluster_duration'] = (clusters['cluster_end'] - clusters['cluster_start']).dt.total_seconds()
        
        total_clusters = len(clusters)
        avg_calls_per_cluster = clusters['calls_in_cluster'].mean()
        largest_cluster = clusters['calls_in_cluster'].max()
        avg_cluster_duration = str(timedelta(seconds=clusters['cluster_duration'].mean()))
        
        # Efficiency Metrics
        daily_stats = user_call_data.groupby('day').agg({
            'call_duration': ['count', 'mean', 'sum'],
            'prev_call_gap': 'mean'
        })
        
        daily_stats.columns = ['calls_per_day', 'avg_duration', 'total_duration', 'avg_gap']
        
        avg_calls_per_day = daily_stats['calls_per_day'].mean()
        avg_gap_seconds = daily_stats['avg_gap'].mean()
        avg_gap_readable = str(timedelta(seconds=avg_gap_seconds))
        most_efficient_day = daily_stats['calls_per_day'].idxmax()
        most_efficient_calls = daily_stats['calls_per_day'].max()
        
        # Peak Time Analysis
        hourly_volume = user_call_data.groupby('hour').size().reset_index(name='calls')
        peak_hours = hourly_volume.nlargest(3, 'calls')
        
        # Correlation Analysis
        correlation_data = user_call_data[['call_duration', 'hour', 'is_weekend']].copy()
        correlation_matrix = correlation_data.corr()
        duration_hour_corr = correlation_matrix.loc['call_duration', 'hour']
        duration_weekend_corr = correlation_matrix.loc['call_duration', 'is_weekend']
        
        # Overall Trend
        daily_calls = user_call_data.resample('D', on='timestamp').size()
        trend_coefficient = np.polyfit(range(len(daily_calls)), daily_calls.values, 1)[0]
        ma30 = daily_calls.rolling(window=30).mean()
        ma30_min = ma30.min()
        ma30_max = ma30.max()
        
        # Duration Distribution Analysis
        user_call_data['duration_category'] = pd.qcut(user_call_data['call_duration'], 
                                                     q=4, 
                                                     labels=['Short', 'Medium', 'Long', 'Very Long'])
        
        time_duration_dist = pd.crosstab(user_call_data['time_of_day'], 
                                        user_call_data['duration_category'], 
                                        normalize='index') * 100
        
        # Anomaly Analysis
        user_call_data['duration_zscore'] = ((user_call_data['call_duration'] - 
                                            user_call_data['call_duration'].mean()) / 
                                           user_call_data['call_duration'].std())
        
        anomalies = user_call_data[abs(user_call_data['duration_zscore']) > 2]
        total_anomalies = len(anomalies)
        anomaly_rate = (total_anomalies / len(user_call_data)) * 100
        avg_anomaly_duration = str(timedelta(seconds=anomalies['call_duration'].mean()))
        
        # Final Summary Statistics
        analysis_period_start = user_call_data['timestamp'].min().date()
        analysis_period_end = user_call_data['timestamp'].max().date()
        total_days_analyzed = daily_calls.shape[0]
        call_volume_trend = "Increasing" if trend_coefficient > 0 else "Decreasing"
        peak_activity_time = busiest_time
        recommended_focus = f"{busiest_time} on {busiest_day}"
        
        # Display enhanced summary statistics in Streamlit
        st.header("📊 Enhanced Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Calls", f"{total_calls:,}")
            st.metric("Weekend Call %", f"{weekend_percentage:.1f}%")
        with col2:
            st.metric("Total Duration", total_duration_readable)
            st.metric("Busiest Day", busiest_day)
        with col3:
            st.metric("Average Duration", avg_duration_readable)
            st.metric("Peak Time", str(busiest_time))
        with col4:
            st.metric("Monthly Growth", f"{call_trend:.1f}%")
            st.metric("Weekend/Weekday Ratio", f"{(weekend_calls/weekday_calls):.2f}")
        
        # New Visualizations
        
        # 1. Call Pattern Analysis
        st.header("🔄 Call Pattern Analysis")
        
        # Create tabs for different views
        pattern_tabs = st.tabs(["Daily Patterns", "Weekly Trends", "Monthly Overview"])
        
        with pattern_tabs[0]:
            # Hourly distribution with rolling average
            hourly_calls = user_call_data.groupby('hour').size()
            rolling_avg = hourly_calls.rolling(window=3, center=True).mean()
            
            fig_hourly = go.Figure()
            fig_hourly.add_bar(x=hourly_calls.index, y=hourly_calls.values, name="Calls per Hour")
            fig_hourly.add_trace(go.Scatter(x=rolling_avg.index, y=rolling_avg.values, 
                                 name="3-Hour Rolling Average", line=dict(color='red')))
            fig_hourly.update_layout(title="Hourly Call Distribution with Rolling Average",
                                   xaxis_title="Hour of Day",
                                   yaxis_title="Number of Calls")
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with pattern_tabs[1]:
            # Weekly pattern heatmap
            weekly_pattern = user_call_data.pivot_table(
                index='day_of_week',
                columns='time_of_day',
                values='call_duration',
                aggfunc='count',
                fill_value=0
            )
            # Ensure day_of_week is ordered correctly
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_pattern = weekly_pattern.reindex(days_order)
            
            fig_weekly = px.imshow(
                weekly_pattern,
                title="Weekly Call Patterns by Time of Day",
                labels=dict(x="Time of Day", y="Day of Week", color="Number of Calls"),
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_weekly, use_container_width=True)
        
        with pattern_tabs[2]:
            # Monthly trend with moving average
            monthly_trend = user_call_data.resample('M', on='timestamp').size()
            
            fig_monthly = go.Figure()
            fig_monthly.add_bar(x=monthly_trend.index, y=monthly_trend.values, name="Monthly Calls")
            fig_monthly.add_trace(go.Scatter(x=monthly_trend.index, 
                                      y=monthly_trend.rolling(window=3).mean(),
                                      name="3-Month Moving Average",
                                      line=dict(color='red')))
            fig_monthly.update_layout(title="Monthly Call Volume Trend",
                                    xaxis_title="Month",
                                    yaxis_title="Number of Calls")
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        # 2. Duration Analysis Dashboard
        st.header("⏱️ Call Duration Analysis")
        
        duration_tabs = st.tabs(["Duration Distribution", "Time-Based Analysis"])
        
        with duration_tabs[0]:
            # Enhanced duration distribution
            fig_duration = make_subplots(rows=2, cols=1,
                                       subplot_titles=("Call Duration Distribution",
                                                     "Cumulative Duration Distribution"))
            
            # Histogram
            fig_duration.add_trace(
                go.Histogram(x=user_call_data['call_duration'],
                           name="Duration Distribution",
                           nbinsx=30),
                row=1, col=1
            )
            
            # Cumulative distribution
            sorted_durations = np.sort(user_call_data['call_duration'])
            cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
            
            fig_duration.add_trace(
                go.Scatter(x=sorted_durations, y=cumulative,
                          name="Cumulative Distribution"),
                row=2, col=1
            )
            
            fig_duration.update_layout(height=800)
            st.plotly_chart(fig_duration, use_container_width=True)
        
        with duration_tabs[1]:
            # Duration by time of day
            avg_duration_by_hour = user_call_data.groupby('hour')['call_duration'].mean()
            
            fig_time_duration = go.Figure()
            fig_time_duration.add_trace(
                go.Scatter(x=avg_duration_by_hour.index,
                          y=avg_duration_by_hour.values,
                          mode='lines+markers',
                          name="Average Duration")
            )
            fig_time_duration.update_layout(title="Average Call Duration by Hour",
                                          xaxis_title="Hour of Day",
                                          yaxis_title="Average Duration (seconds)")
            st.plotly_chart(fig_time_duration, use_container_width=True)
        
        # 3. Advanced Pattern Recognition
        st.header("🎯 Advanced Pattern Recognition")
        
        # Visualize clusters
        fig_clusters = px.scatter(clusters.reset_index(),
                                x='cluster_start',
                                y='calls_in_cluster',
                                size='total_duration',
                                hover_data=['cluster_duration'],
                                title="Call Clusters Analysis")
        
        st.plotly_chart(fig_clusters, use_container_width=True)
        
        # Display Cluster Insights
        st.subheader("Cluster Insights")
        st.write(f"- **Total Clusters:** {total_clusters}")
        st.write(f"- **Average Calls per Cluster:** {avg_calls_per_cluster:.1f}")
        st.write(f"- **Largest Cluster:** {largest_cluster} calls")
        st.write(f"- **Average Cluster Duration:** {avg_cluster_duration}")
        
        # 4. Comparative Analysis
        st.header("🔍 Comparative Analysis")
        
        # Compare weekday vs weekend patterns
        weekday_pattern = user_call_data[~user_call_data['is_weekend']].groupby('hour').size().reset_index(name='calls')
        weekend_pattern = user_call_data[user_call_data['is_weekend']].groupby('hour').size().reset_index(name='calls')
        
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Scatter(x=weekday_pattern['hour'],
                                          y=weekday_pattern['calls'],
                                          name="Weekday Calls",
                                          mode='lines+markers'))
        fig_comparison.add_trace(go.Scatter(x=weekend_pattern['hour'],
                                          y=weekend_pattern['calls'],
                                          name="Weekend Calls",
                                          mode='lines+markers'))
        
        fig_comparison.update_layout(title="Weekday vs Weekend Call Patterns",
                                   xaxis_title="Hour of Day",
                                   yaxis_title="Number of Calls")
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # 5. Efficiency Metrics
        st.header("📈 Efficiency Metrics")
        
        fig_efficiency = make_subplots(rows=2, cols=1,
                                     subplot_titles=("Daily Call Volume",
                                                   "Average Call Gap"))
        
        fig_efficiency.add_trace(
            go.Scatter(x=daily_stats.index,
                      y=daily_stats['calls_per_day'],
                      mode='lines',
                      name="Calls per Day"),
            row=1, col=1
        )
        
        fig_efficiency.add_trace(
            go.Scatter(x=daily_stats.index,
                      y=daily_stats['avg_gap'],
                      mode='lines',
                      name="Average Gap (seconds)"),
            row=2, col=1
        )
        
        fig_efficiency.update_layout(height=800)
        st.plotly_chart(fig_efficiency, use_container_width=True)
        
        # Display Efficiency Insights
        st.subheader("Efficiency Insights")
        st.write(f"- **Average Calls per Day:** {avg_calls_per_day:.1f}")
        st.write(f"- **Average Gap Between Calls:** {avg_gap_readable}")
        st.write(f"- **Most Efficient Day:** {most_efficient_day} ({most_efficient_calls} calls)")
        
        # 6. Peak Time Analysis
        st.header("⏰ Peak Time Analysis")
        
        # Calculate and display peak hours
        st.subheader("Top 3 Busiest Hours")
        for index, row in peak_hours.iterrows():
            st.write(f"- **Hour {int(row['hour']):02d}:00:** {row['calls']} calls")
        
        # Display Peak Time Scatter Plot
        fig_peaks = px.scatter(
            hourly_volume,
            x='hour',
            y='calls',
            color='calls',
            title="Call Volume by Hour",
            labels={'hour': 'Hour of Day', 'calls': 'Number of Calls'},
            color_continuous_scale='Blues'
        )
        
        st.plotly_chart(fig_peaks, use_container_width=True)
        
        # 7. Call Duration Relationship Analysis
        st.header("🔗 Call Duration Relationships")
        
        # Create correlation matrix
        correlation_matrix = correlation_data.corr()
        
        fig_corr = px.imshow(correlation_matrix,
                            title="Correlation Matrix of Call Metrics",
                            labels=dict(color="Correlation Coefficient"),
                            text_auto=True)
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Display Correlation Insights
        st.subheader("Correlation Insights")
        st.write(f"- **Duration-Hour Correlation:** {duration_hour_corr:.3f}")
        st.write(f"- **Duration-Weekend Correlation:** {duration_weekend_corr:.3f}")
        
        # 8. Call Volume Forecasting
        st.header("🔮 Call Volume Trends")
        
        # Calculate daily moving averages
        ma7 = daily_calls.rolling(window=7).mean()
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=daily_calls.index, y=daily_calls.values,
                                        mode='lines', name='Daily Calls',
                                        line=dict(color='lightgrey')))
        fig_forecast.add_trace(go.Scatter(x=ma7.index, y=ma7.values,
                                        mode='lines', name='7-day MA',
                                        line=dict(color='blue')))
        fig_forecast.add_trace(go.Scatter(x=ma30.index, y=ma30.values,
                                        mode='lines', name='30-day MA',
                                        line=dict(color='red')))
        
        fig_forecast.update_layout(title='Call Volume Trends with Moving Averages',
                                 xaxis_title='Date',
                                 yaxis_title='Number of Calls')
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Display Trend Insights
        st.subheader("Trend Insights")
        st.write(f"- **Overall Trend Coefficient:** {trend_coefficient:.3f} calls/day")
        st.write(f"- **30-day Moving Average Range:** {ma30_min:.1f} - {ma30_max:.1f} calls")
        
        # 9. Call Duration Distribution by Time Period
        st.header("📊 Duration Distribution Analysis")
        
        fig_duration_dist = px.bar(time_duration_dist.reset_index(),
                                 x='time_of_day',
                                 y=['Short', 'Medium', 'Long', 'Very Long'],
                                 title="Call Duration Distribution by Time of Day",
                                 labels={'value': 'Percentage', 'time_of_day': 'Time of Day'},
                                 barmode='stack',
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
        
        st.plotly_chart(fig_duration_dist, use_container_width=True)
        
        # Display Duration Distribution Insights
        st.subheader("Duration Distribution Insights")
        for period in time_duration_dist.index:
            st.write(f"**{period}:**")
            for category in time_duration_dist.columns:
                st.write(f"- {category}: {time_duration_dist.loc[period, category]:.1f}%")
            st.write("")  # Add space
        
        # 10. Call Pattern Anomaly Detection
        st.header("🚨 Pattern Anomaly Analysis")
        
        fig_anomalies = go.Figure()
        
        # Plot all calls
        fig_anomalies.add_trace(go.Scatter(x=user_call_data['timestamp'],
                                         y=user_call_data['call_duration'],
                                         mode='markers',
                                         name='Normal Calls',
                                         marker=dict(color='blue', size=8, opacity=0.6)))
        
        # Highlight anomalies
        fig_anomalies.add_trace(go.Scatter(x=anomalies['timestamp'],
                                         y=anomalies['call_duration'],
                                         mode='markers',
                                         name='Anomalies',
                                         marker=dict(color='red', size=12, symbol='star')))
        
        fig_anomalies.update_layout(title='Call Duration Anomalies',
                                  xaxis_title='Time',
                                  yaxis_title='Call Duration (seconds)')
        
        st.plotly_chart(fig_anomalies, use_container_width=True)
        
        # Display Anomaly Insights
        
        # Final Summary
        st.header("📋 Key Findings Summary")
        
        st.write(f"""
        ### Pattern Highlights
        * Most active time period is during **{busiest_time}**
        * **{busiest_day}** shows consistently higher call volumes
        * Average call duration peaks at **{int(peak_hours.iloc[0]['hour'])}:00 hours**
        
        ### Efficiency Metrics
        * Daily call volume varies from **{daily_stats['calls_per_day'].min():.1f}** to **{daily_stats['calls_per_day'].max():.1f}** calls
        * **{(anomaly_rate):.1f}%** of calls occur during peak hours
        * Average gap between calls: **{avg_gap_readable}**
        
        """)
        
        # Display Final Summary Statistics
        st.subheader("Final Summary Statistics")
        st.write(f"- **Analysis Period:** {analysis_period_start} to {analysis_period_end}")
        st.write(f"- **Total Days Analyzed:** {total_days_analyzed}")
        st.write(f"- **Average Daily Call Volume:** {avg_calls_per_day:.1f}")
        st.write(f"- **Call Volume Trend:** {call_volume_trend}")
        st.write(f"- **Peak Activity Time:** {peak_activity_time}")

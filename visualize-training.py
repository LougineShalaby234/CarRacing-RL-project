import streamlit as st
import json
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

def load_metrics(metrics_file='training_metrics/training_metrics.json'):
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def plot_reward_history(metrics):
    st.subheader("Total Reward Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=metrics['rewards']['total'],
        mode='lines',
        name='Total Reward',
        line=dict(color='#2E86C1')
    ))
    fig.update_layout(
        xaxis_title="Episode",
        yaxis_title="Reward",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_reward_components(metrics):
    st.subheader("Reward Components")
    if metrics['rewards']['components']:
        fig = go.Figure()
        components = metrics['rewards']['components'][0].keys()
        colors = ['#2E86C1', '#28B463', '#D35400', '#8E44AD', '#C0392B']
        
        for component, color in zip(components, colors):
            values = [episode[component] for episode in metrics['rewards']['components']]
            # Calculate moving average
            window = 10
            moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
            
            fig.add_trace(go.Scatter(
                y=moving_avg,
                mode='lines',
                name=component.capitalize(),
                line=dict(color=color)
            ))
        
        fig.update_layout(
            xaxis_title="Episode",
            yaxis_title="Reward Component Value",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_weight_distributions(metrics):
    st.subheader("Network Weights Distribution")
    if metrics['weights']:
        layers = list(metrics['weights'][0].keys())
        selected_layer = st.selectbox("Select Layer", layers)
        
        fig = go.Figure()
        stats = ['mean', 'std', 'max', 'min']
        colors = ['#2E86C1', '#28B463', '#D35400', '#8E44AD']
        
        for stat, color in zip(stats, colors):
            values = [episode[selected_layer][stat] for episode in metrics['weights']]
            fig.add_trace(go.Scatter(
                y=values,
                mode='lines',
                name=stat.capitalize(),
                line=dict(color=color)
            ))
        
        fig.update_layout(
            xaxis_title="Episode",
            yaxis_title="Weight Value",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_action_distribution(metrics):
    st.subheader("Action Distribution Over Time")
    fig = go.Figure()
    actions = ['No Action', 'Left', 'Right', 'Gas', 'Brake']
    colors = ['#2E86C1', '#28B463', '#D35400', '#8E44AD', '#C0392B']
    
    for i, (action, color) in enumerate(zip(actions, colors)):
        values = [dist[i] for dist in metrics['action_distribution']]
        # Calculate moving average
        window = 10
        moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
        
        fig.add_trace(go.Scatter(
            y=moving_avg,
            mode='lines',
            name=action,
            line=dict(color=color)
        ))
    
    fig.update_layout(
        xaxis_title="Episode",
        yaxis_title="Action Probability",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_training_parameters(metrics):
    st.subheader("Training Parameters")
    fig = go.Figure()
    params = ['epsilon', 'loss', 'avg_q_value']
    colors = ['#2E86C1', '#28B463', '#D35400']
    
    for param, color in zip(params, colors):
        values = [episode[param] for episode in metrics['parameters']]
        # Calculate moving average for smoothing
        window = 10
        moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
        
        fig.add_trace(go.Scatter(
            y=moving_avg,
            mode='lines',
            name=param.replace('_', ' ').title(),
            line=dict(color=color)
        ))
    
    fig.update_layout(
        xaxis_title="Episode",
        yaxis_title="Parameter Value",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="DQN Training Dashboard", layout="wide")
    st.title("DQN Training Visualization Dashboard")
    
    metrics = load_metrics()
    if metrics is None:
        st.error("No training metrics found. Please run training first.")
        return
    
    # Create tabs for different visualizations
    tabs = st.tabs(["Rewards", "Weights", "Policy", "Parameters"])
    
    with tabs[0]:  # Rewards
        col1, col2 = st.columns(2)
        with col1:
            plot_reward_history(metrics)
        with col2:
            plot_reward_components(metrics)
    
    with tabs[1]:  # Weights
        plot_weight_distributions(metrics)
    
    with tabs[2]:  # Policy
        plot_action_distribution(metrics)
    
    with tabs[3]:  # Parameters
        plot_training_parameters(metrics)

    # Add auto-refresh button
    if st.button('Refresh Data'):
        st.experimental_rerun()

if __name__ == "__main__":
    main()

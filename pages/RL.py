import streamlit as st
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from yahoo_fin import stock_info as si
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import plotly.graph_objects as go
import yfinance as yf

class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, window_size=30):
        super(PortfolioEnv, self).__init__()
        self.df = df
        self.n_assets = df.shape[1]
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.episode_return = 0

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_assets * self.window_size + self.n_assets,),
            dtype=np.float32
        )

        self.current_step = self.window_size
        self.balance = initial_balance
        self.allocations = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = [initial_balance]

    def _get_observation(self):
        obs = self.df.iloc[self.current_step - self.window_size : self.current_step].values.flatten()
        return np.append(obs, self.allocations)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.allocations = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = [self.initial_balance]
        self.episode_return = 0
        return self._get_observation(), {}

    def step(self, action):
        self.allocations = np.exp(action) / np.sum(np.exp(action))
        daily_returns = self.df.iloc[self.current_step].values
        new_value = self.balance * np.dot(self.allocations, (1 + daily_returns))
        reward = np.log(new_value / self.balance)
        self.episode_return += reward
        
        self.balance = new_value
        self.portfolio_value.append(self.balance)
        self.current_step += 1

        terminated = self.current_step >= len(self.df)
        truncated = False
        info = {}
        
        if terminated or truncated:
            info['episode'] = {'r': self.episode_return, 'l': self.current_step - self.window_size}

        return self._get_observation(), reward, terminated, truncated, info

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        
    def _on_step(self):
        for idx, done in enumerate(self.locals['dones']):
            if done and 'episode' in self.locals['infos'][idx]:
                self.episode_rewards.append(self.locals['infos'][idx]['episode']['r'])
                st.session_state.rewards = self.episode_rewards.copy()
        return True

def main():
    st.title("Reinforcement Learning Portfolio Optimizer")
    
    with st.sidebar:
        st.header("Configuration")
        tickers = st.text_input("Assets (comma-separated)", "AAPL,MSFT,SPY").upper().split(',')
        start_date = st.date_input("Start Date", pd.to_datetime("2010-01-01"))
        end_date = st.date_input("End Date", pd.to_datetime("2020-12-31"))
        initial_balance = st.number_input("Initial Balance ($)", 1000, 1000000, 10000)
        window_size = st.slider("Observation Window Size", 10, 100, 30)
        train_timesteps = st.slider("Training Timesteps", 1000, 100000, 20000, step=1000)

    if 'env' not in st.session_state:
        st.session_state.env = None
    if 'model' not in st.session_state:
        st.session_state.model = None

    if st.button("Initialize/Reload Environment"):
        try:
            # Download data with yfinance
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                group_by='ticker'
            )
            
            # Check if data exists
            if data.empty:
                st.error("No data found. Check tickers and dates.")
                return
            
            # Extract CLOSE prices (not adjusted close)
            close_prices = data.xs('Close', level=1, axis=1).copy()  # <-- Key fix here
            close_prices.dropna(axis=1, how='all', inplace=True)  # Remove empty columns
            
            if close_prices.empty:
                st.error("No valid closing prices found.")
                return
            
            # Calculate returns
            returns = close_prices.pct_change().shift(-1).dropna()
            
            # Initialize environment
            st.session_state.env = PortfolioEnv(
                df=returns,
                initial_balance=initial_balance,
                window_size=window_size
            )
            st.success(f"Environment initialized with: {', '.join(close_prices.columns)}!")
            st.line_chart(close_prices)
            
        except Exception as e:
            st.error(f"Data error: {str(e)}")

    if st.session_state.env is not None:
        if st.button("Train Model"):
            with st.spinner(f"Training for {train_timesteps} timesteps..."):
                try:
                    model = PPO(
                        'MlpPolicy',
                        st.session_state.env,
                        verbose=0
                    )
                    callback = RewardCallback()
                    model.learn(total_timesteps=train_timesteps, callback=callback)
                    st.session_state.model = model
                    st.success("Training completed!")
                    
                    if 'rewards' in st.session_state and st.session_state.rewards:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(range(len(st.session_state.rewards))),
                            y=st.session_state.rewards,
                            mode='lines+markers',
                            name='Training Reward',
                            hovertemplate='<b>Episode %{x}</b><br>Return: %{y:.2f}<extra></extra>'
                        ))
                        fig.update_layout(
                            title='Training Progress',
                            xaxis_title='Episode',
                            yaxis_title='Cumulative Return'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No training rewards recorded")
                        
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")

    if st.session_state.model is not None:
        if st.button("Run Portfolio Simulation"):
            portfolio_values = []
            obs, info = st.session_state.env.reset()
            done = False
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while not done:
                action, _ = st.session_state.model.predict(obs)
                obs, _, terminated, truncated, _ = st.session_state.env.step(action)
                done = terminated or truncated
                portfolio_values.append(st.session_state.env.balance)
                
                progress = st.session_state.env.current_step / len(st.session_state.env.df)
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Step {st.session_state.env.current_step} | Value: ${st.session_state.env.balance:,.2f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(portfolio_values))),
                y=portfolio_values,
                mode='lines+markers',
                name='Portfolio Value',
                hovertemplate='<b>Step %{x}</b><br>Value: $%{y:,.2f}<extra></extra>'
            ))
            fig.update_layout(
                title='Portfolio Performance',
                xaxis_title='Time Steps',
                yaxis_title='Portfolio Value ($)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            final_value = portfolio_values[-1]
            total_return = (final_value / initial_balance - 1) * 100
            st.metric("Final Portfolio Value", f"${final_value:,.2f}", 
                     f"{total_return:.2f}% Return")

if __name__ == "__main__":
    main()
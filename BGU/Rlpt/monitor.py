from multiprocessing import Process
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from collections import deque
import plotly.graph_objects as go
from threading import Event, Thread
import time

class Monitor:
    def __init__(self, window_size = 2000,port='8052',x_label='x', y_labels=[]):    
        self.window_size = window_size
        self.port = port
        self.x_data = deque(maxlen=window_size)
        self.x_label = x_label
        self.y_labels = y_labels
        self.y_datas = [deque(maxlen=window_size) for _ in y_labels]
        
        # Placeholder for external data
        self.current_x = 0
        self.current_ys = [0] * len(y_labels)
        self.process = None
        # self.exit_event = Event()  # Use an event to signal exit

        # self.app  = dash.Dash(__name__)

        # # Define the layout with the graph and a hidden interval component for updates
        # self.app.layout = html.Div([
        #     dcc.Graph(id='live-graph'),
        #     dcc.Interval(id='interval-component', interval=100, n_intervals=0)  # Update every 100ms
        # ])

        # # Update the graph at regulary intervals
        # @self.app.callback(
        #     Output('live-graph', 'figure'),
        #     [Input('interval-component', 'n_intervals')]
        # )
        # def update_graph(n_intervals):
        #     # Generate new data and add it to the deque
        #     self.x_data.append(self.current_x)
        #     for i, y_data in enumerate(self.y_datas):
        #         y_data.append(self.current_ys[i])
                
        #     # Create figure
        #     fig = go.Figure()
            
        #     # self.x_data.append(x)
        #     for i, y_data in enumerate(self.y_datas):
        #         fig.add_trace(go.Scatter(x=list(self.x_data), y=list(y_data), mode='lines', name=self.y_labels[i]))
            
        #     # fig.add_trace(go.Scatter(x=list(x_data), y=list(y2_data), mode='lines', name='y2'))
        #     # fig.add_trace(go.Scatter(x=list(x_data), y=list(y3_data), mode='lines', name='y3'))

        #     # Set layout details
        #     fig.update_layout(
        #         # title="Real-time Data for y1, y2, y3",
        #         xaxis_title=self.x_label,
        #         yaxis_title=str(self.y_labels),
        #         xaxis=dict(range=[max(0, self.x_data[-1] - window_size), self.x_data[-1]]),
        #         yaxis=dict(range=[-500, 100])
        #     )

        #     return fig
        
        
    
    def update_data(self, x, ys):
        """
        Update the data to be plotted.

        Args:
            x (int or float): New x value.
            ys (list): List of new y values, one for each trace.
        """
        self.current_x = x
        self.current_ys = ys
        
    def run_app(self):
    
        
        self.app  = dash.Dash(__name__)

        # Define the layout with the graph and a hidden interval component for updates
        self.app.layout = html.Div([
            dcc.Graph(id='live-graph'),
            dcc.Interval(id='interval-component', interval=100, n_intervals=0)  # Update every 100ms
        ])

        # Update the graph at regulary intervals
        @self.app.callback(
            Output('live-graph', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_graph(n_intervals):
            # Generate new data and add it to the deque
            self.x_data.append(self.current_x)
            for i, y_data in enumerate(self.y_datas):
                y_data.append(self.current_ys[i])
            
            # Create figure
            fig = go.Figure()
            
            # self.x_data.append(x)
            for i, y_data in enumerate(self.y_datas):
                fig.add_trace(go.Scatter(x=list(self.x_data), y=list(y_data), mode='lines', name=self.y_labels[i]))
            
            # fig.add_trace(go.Scatter(x=list(x_data), y=list(y2_data), mode='lines', name='y2'))
            # fig.add_trace(go.Scatter(x=list(x_data), y=list(y3_data), mode='lines', name='y3'))

            # Set layout details
            fig.update_layout(
                # title="Real-time Data for y1, y2, y3",
                xaxis_title=self.x_label,
                yaxis_title=str(self.y_labels),
                xaxis=dict(range=[max(0, self.x_data[-1] - self.window_size), self.x_data[-1]]),
                yaxis=dict(range=[-100, 100])
            )

            return fig
        
        self.app.run_server(debug=False, use_reloader=False,port=self.port)
    
    def start(self):
        # Start Dash app in a thread and check for the exit flag
        # self.thread = Thread(target=self.run_app)
        self.process = Process(target=self.run_app)
        self.process.start()
        # self.thread.start()
        
    def stop(self):
        # Set the event flag to signal exit
        # self.exit_event.set()
        # # Close Dash server after the next tick
        # self.app._server.shutdown()
        if self.process:
            self.process.terminate()
            self.process.join()
        
# # Initialize time and data arrays with a fixed window size of 2000

# window_size = 2000
# x_data = deque(maxlen=window_size)
# y1_data = deque(maxlen=window_size)
# y2_data = deque(maxlen=window_size)
# y3_data = deque(maxlen=window_size)

# # Initialize the Dash app
# app = dash.Dash(__name__)

# # Define the layout with the graph and a hidden interval component for updates
# app.layout = html.Div([
#     dcc.Graph(id='live-graph'),
#     dcc.Interval(id='interval-component', interval=100, n_intervals=0)  # Update every 100ms
# ])

# # Update the graph at regular intervals
# @app.callback(
#     Output('live-graph', 'figure'),
#     [Input('interval-component', 'n_intervals')]
# )
# def update_graph(n):
#     # Generate new data and add it to the deque
#     x_data.append(n)
#     y1_data.append(np.random.randn())
#     y2_data.append(np.random.randn())
#     y3_data.append(np.random.randn())

#     # Create figure
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=list(x_data), y=list(y1_data), mode='lines', name='y1'))
#     fig.add_trace(go.Scatter(x=list(x_data), y=list(y2_data), mode='lines', name='y2'))
#     fig.add_trace(go.Scatter(x=list(x_data), y=list(y3_data), mode='lines', name='y3'))

#     # Set layout details
#     fig.update_layout(
#         title="Real-time Data for y1, y2, y3",
#         xaxis_title="Time (steps)",
#         yaxis_title="Values",
#         xaxis=dict(range=[max(0, n - window_size), n]),
#         yaxis=dict(range=[-10, 10])
#     )

#     return fig

# # Run the app in a separate thread to keep it non-blocking
# def run_app():
#     app.run_server(debug=False, use_reloader=False,port='8052')

# # Start Dash app in a background thread
# thread = Thread(target=run_app)
# thread.start()

# # Simulate data collection in the main thread
# for _ in range(10000):  # Simulate 10,000 steps
#     time.sleep(0.0001)  # Adjust this sleep interval as needed

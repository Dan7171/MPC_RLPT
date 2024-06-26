# Shared dictionary
# shared_data = {'a': 0, 'b': 0, 'c': 0}
from BGU.Rlpt.DebugTools.storm_tools import RealWorldState
from BGU.Rlpt.Classes.CostTerm import CostTerm
import threading
lock = threading.Lock()

# Function to update the shared dictionary (this would be your data generation logic)
def update_costs():
    while True:
        with lock:
            update_()
        #     shared_data['a'] = np.random.rand() * 100
        #     shared_data['b'] = np.random.rand() * 100
        #     shared_data['c'] = np.random.rand() * 100
        # time.sleep(0.5)

# Function to read and display the shared data
def read_and_display_data():
    plt.ion()
    fig, ax = plt.subplots()
    bars = ax.bar(['a', 'b', 'c'], [0, 0, 0], color='green')

    # Set up the color map
    norm = Normalize(vmin=0, vmax=100)
    cmap = plt.get_cmap('RdYlGn_r')
    sm = ScalarMappable(norm=norm, cmap=cmap)

    while True:
        with lock:
            a = shared_data['a']
            b = shared_data['b']
            c = shared_data['c']

        values = [a, b, c]
        for bar, value in zip(bars, values):
            bar.set_height(value)
            bar.set_color(sm.to_rgba(value))

        ax.set_ylim(0, 100)  # Ensure the y-axis always goes from 0 to 100

        plt.draw()
        plt.pause(0.01)

        time.sleep(0.01)



# Start the data updating thread
update_thread = threading.Thread(target=update_shared_data)
update_thread.daemon = True
update_thread.start()

# Start the data reading and displaying thread
display_thread = threading.Thread(target=read_and_display_data)
display_thread.daemon = True
display_thread.start()

# # Keep the main thread alive
# while True:
#     time.sleep(1)


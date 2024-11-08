import matplotlib.pyplot as plt
import pandas as pd

def plot_data(final_output_file, diff, x_axis, y_axis_actual, y_axis_predicted, title):
    plot_data = pd.read_csv(final_output_file)
    data = plot_data[::diff]
    # Create a line plot with cap_id on the x-axis
    plt.figure(figsize=(10, 5))
    # Plot each RWA measure on the same graph
    # plt.plot(data[x_axis], data['dtm_rwa'], label='DTM RWA', marker='o')
    plt.plot(data[x_axis], data[y_axis_predicted], label=y_axis_predicted, marker='o')
    plt.plot(data[x_axis], data[y_axis_actual], label=y_axis_actual, marker='o')

    # Add labels and title
    plt.xlabel(x_axis)
    plt.ylabel(y_axis_actual+' Values')
    plt.title(title)
    # Add a legend to differentiate the lines
    plt.legend()
    # Optional: Rotate x-axis labels if needed
    plt.xticks(rotation=45)
    # Show grid lines for readability
    plt.grid(True)
    # Display the plot
    plt.tight_layout()
    plt.show()
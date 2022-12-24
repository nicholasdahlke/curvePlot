import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
import csv
import scipy

############################### Constants ###############################
c1 = 0.643942 / 1000 #
c2 = 0.654876 / 1000 # Kennwerte der Kanäle des Kraftsensors
c3 = 0.677194 / 1000 #

amplifier_sensitivity = 0.5 / 1000 # mv/V
amplifier_output_min_max = 10 # V Output Range of the amplifier
u_speise = 5  # Speisespannung des Kraftsensors - 5V
f_nom = 20  # Nennkraft 20N

chord_length = 0.0493  # Chordlength des Flügels
wing_span = 0.15  # Flügelspanne
surface_area = chord_length * wing_span  # Fläche des Flügels
air_density = 1.225  # Luftdichte
flow_speed = 10  # Freistromgeschwindigkeit

angle_filter_length = 10  # Fensterbreite des Moving Median Winkel
force_filter_length =  10000 # Fensterbreite des Moving Average Kräfte

naca = "0018"
re = 33000
ncrit = 5
iter = 5000

sweep_sim_angle = 20
sweep_sim_step = 0.5

use_zero_signal = True
############################### Convert the LabVIEW Waveform Timestamps into a usable Format ###########################
def convert_timestamp(timestamp):
    print("Convert timestamp " + timestamp)
    timestamp = timestamp[11:len(timestamp)].strip()
    hours = int(timestamp[0:2]) * 3600
    minutes = int(timestamp[3:5]) * 60
    seconds = float(timestamp[6:len(timestamp)].replace(",", "."))
    print(hours + minutes + seconds)
    return float(hours + minutes + seconds)

convert_timestamp_vec = np.vectorize(convert_timestamp, otypes=[float])


########################################################################################################################

############################### Data Input Functions ###############################
def read_angle_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=";")
        for row in reader:
            if row[0] != "":
                row[1] = float(row[1].replace(",", "."))
                data.append(row)
    data_np = np.array(data)
    data_np[:,0] = convert_timestamp_vec(data_np[:,0])
    print("Read angle data of length")
    print(len(data))
    return data_np.astype(float)


def read_force_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=";")
        for row in reader:
            if row[0] != "":
                row[1] = float(row[1].replace(",", "."))
                row[2] = float(row[2].replace(",", "."))
                row[3] = float(row[3].replace(",", "."))
                data.append(row)
    data_np = np.array(data)
    data_np[:,0] = convert_timestamp_vec(data_np[:,0])
    print("Read force data of length")
    print(len(data))
    return data_np.astype(float)


def read_sim_data(filename):

    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            for i in range(9):
                row[i] = float(row[i])
            data.append(row)
    print("Read simulation data of length")
    print(len(data))
    return np.array(data)


#####################################################################################

#####################################################################################
# Use XFoil to generate a reference polar
def gen_xfoil_data():
    xfoil_save_file = "xfoil_naca" + naca + "re" + str(re) + "ncrit" + str(ncrit) + ".dat" # File, which the polar is saved to

    xfoil_inp = "NACA " + naca + "\n"
    xfoil_inp += "GDES\n"
    xfoil_inp += "\n"
    xfoil_inp += "OPER\n"
    xfoil_inp += "VPAR\n"
    xfoil_inp += "N\n"
    xfoil_inp += str(ncrit) + "\n"
    xfoil_inp += "\n"
    xfoil_inp += "RE\n"
    xfoil_inp += str(re) + "\n"
    xfoil_inp += "ITER\n"
    xfoil_inp += str(iter) + "\n"
    xfoil_inp += "VISC\n"
    xfoil_inp += "PACC\n"
    xfoil_inp += xfoil_save_file + "\n"
    xfoil_inp += "\n"
    xfoil_inp += "ASEQ " + str(-sweep_sim_angle) + " " + str(sweep_sim_angle) + " " + str(sweep_sim_step) + "\n"
    xfoil_inp += "\n"
    xfoil_inp += "QUIT"
    print(xfoil_inp)
    xfoil_inp = xfoil_inp.encode('ascii') #

    xfoil_p = sp.Popen(["xfoil"], stdout=sp.PIPE, stdin=sp.PIPE, stderr=sp.PIPE) # open an XFoil subprocess
    xfoil_out = xfoil_p.communicate(input=xfoil_inp) # send xfoil_inp via stdin

    clean_p = sp.Popen(["/home/nicholas/PycharmProjects/curvePlot/clean_sim_file.sh", xfoil_save_file.encode('ascii')]) # cleanup the results for use with numpy
    return xfoil_save_file + ".csv"
#####################################################################################


#####################################################################################
# Sync the force and angle data using the timestamps
def sync_merge_data(angles, forces):
    start_index = 0
    for i, force in enumerate(forces):
        if force[0] >= angles[0][0]:
            start_index = i + 0 # set index offset here, to skip wrong data
            break

    forces = forces[start_index:-1] # set end offset here
    angles_interp = np.interp(forces[:, 0], angles[:, 0], angles[:, 1], left=np.nan, right=np.nan)
    return np.column_stack((forces[:, 0], angles_interp, forces[:, 1], forces[:, 2], forces[:, 3]))
#####################################################################################

#####################################################################################
# Convert the voltage signal from the sensor to a force signal
def convert_voltage_to_force(u, f_kenn, c, u_speise, amplifier_sensitivity):
    voltage = (u_speise / amplifier_sensitivity) * u
    return voltage * u_speise * c / f_kenn
#####################################################################################

#####################################################################################
# Calculate Lift and Drag Coefficient
def calculate_lift_coefficient(lift_force, fluid_density, flow_speed, surface_area):
    dynamic_pressure = (fluid_density/2)*flow_speed**2
    return lift_force / (dynamic_pressure * surface_area)


def calculate_drag_coefficient(drag_force, fluid_density, flow_speed, surface_area):
    return (2*drag_force)/(fluid_density * flow_speed**2 *surface_area)

#####################################################################################

#####################################################################################
# Create vectorized functions numpy can use efficiently
convert_voltage_to_force_vec = np.vectorize(convert_voltage_to_force)
calculate_lift_coefficient_vec = np.vectorize(calculate_lift_coefficient)
calculate_drag_coefficient_vec = np.vectorize(calculate_drag_coefficient)
#####################################################################################

#####################################################################################
# Read the angle and force data into numpy arrays
angles = read_angle_data("Real/NACA 0018/5_10ms/serial.csv")
forces = read_force_data("Real/NACA 0018/5_10ms/daq.csv")

if use_zero_signal:
    print("Using zero signal")
    zero = read_force_data("Real/NACA 0018/5_10ms/zero_signal/daq.csv")
#####################################################################################

#####################################################################################
# Merge angle and force data and save them to a file
merged = sync_merge_data(angles, forces)
np.save("save.npy", merged)
#####################################################################################

#####################################################################################
#Allow quick reloading of the data
merged = np.load("save.npy", allow_pickle=True)
#####################################################################################


#####################################################################################
# Convert all the voltages to forces
f1 = convert_voltage_to_force_vec(merged[:, 2], f_nom, c1, u_speise, amplifier_sensitivity)
print("f1 converted")

f2 = convert_voltage_to_force_vec(merged[:, 3], f_nom, c2, u_speise, amplifier_sensitivity)
print("f2 converted")

f3 = convert_voltage_to_force_vec(merged[:, 4], f_nom, c3, u_speise, amplifier_sensitivity)
print("f3 converted")

if use_zero_signal:
    f1_zero = convert_voltage_to_force_vec(zero[:, 1], f_nom, c1, u_speise, amplifier_sensitivity)
    print("f1 zero converted")

    f2_zero = convert_voltage_to_force_vec(zero[:, 2], f_nom, c2, u_speise, amplifier_sensitivity)
    print("f2 zero converted")

    f3_zero = convert_voltage_to_force_vec(zero[:, 3], f_nom, c3, u_speise, amplifier_sensitivity)
    print("f3 zero converted")
#####################################################################################

#####################################################################################
# Calculate the zero signal on all three axes
f1_offset = np.average(f1_zero)
f2_offset = np.average(f2_zero)
f3_offset = np.average(f3_zero)

print("f1 offset: ", f1_offset)
print("f2 offset: ", f2_offset)
print("f3 offset: ", f3_offset)
#####################################################################################

#####################################################################################
# Apply the offset correction to all three channels
f1 -= f1_offset
f2 -= f2_offset
f3 -= f3_offset
#####################################################################################


#####################################################################################
# Smooth the signals using a moving average filter with the window size force_filter_length
f1_smooth = np.convolve(f1, np.ones(force_filter_length) / force_filter_length, mode="same")
f2_smooth = np.convolve(f2, np.ones(force_filter_length) / force_filter_length, mode="same")
f3_smooth = np.convolve(f3, np.ones(force_filter_length) / force_filter_length, mode="same")
#####################################################################################

#####################################################################################
# For clearer naming
lift_force = f1_smooth
drag_force = f3_smooth
#####################################################################################

#####################################################################################
# Calculate lift and drag coefficients
lift_coefficient = calculate_lift_coefficient_vec(lift_force, air_density, flow_speed, surface_area)
#diff = lift_coefficient.max() + lift_coefficient.min()  #
#lift_coefficient = lift_coefficient - diff / 2          # Shift the lift curve if necessary

drag_coefficient = calculate_drag_coefficient_vec(drag_force, air_density, flow_speed, surface_area)
#####################################################################################

#####################################################################################
# Smooth the angle data using a moving median to remove outliers
angle_smooth = scipy.ndimage.median_filter(merged[:, 1], angle_filter_length, mode='reflect')
#####################################################################################

#####################################################################################
# Read in the simulated data
xfoil_save_file = gen_xfoil_data()
sim_data = read_sim_data(xfoil_save_file)
#####################################################################################

#####################################################################################
# Calculate C_L/C_D
lift_div_drag = lift_coefficient/drag_coefficient
sim_lift_div_drag = sim_data[:,1] / sim_data[:,2]
#####################################################################################

#####################################################################################
# Create a 2x2 Grid of subplots
fig, axs = plt.subplots(2,2)
#####################################################################################

#####################################################################################
# Plot all the smoothed force signals
f1_plot = axs[0,0].plot(angle_smooth, f1_smooth, label="$F_1$")
f2_plot = axs[0,0].plot(angle_smooth, f2_smooth, label="$F_2$")
f3_plot = axs[0,0].plot(angle_smooth, f3_smooth, label="$F_3$")
axs[0,0].set_ylabel("$F$")
axs[0,0].set_title(r"$F$ vs $\alpha$")
axs[0,0].legend()
#####################################################################################

#####################################################################################
# Plot measured and simulated drag coefficient
drag_plot = axs[0,1].plot(angle_smooth, drag_coefficient, label="$C_D$", color="blue")
drag_sim_plot = axs[0,1].plot(sim_data[:,0], sim_data[:,2], label="XFoil $C_D$", linestyle="--", color="blue")
axs[0,1].set_ylabel("$C_D$")
axs[0,1].set_title(r"$C_D$ vs $\alpha$")
axs[0,1].legend()
#####################################################################################

#####################################################################################
# Plot measured and simulated lift coefficient
lift_plot = axs[1,0].plot(angle_smooth, lift_coefficient, label="$C_L$", color="orange")
lift_sim_plot = axs[1,0].plot(sim_data[:, 0], sim_data[:, 1], label="XFoil $C_L$", linestyle="--", color="orange")
axs[1,0].set_ylabel("$C_L$")
axs[1,0].set_title(r"$C_L$ vs $\alpha$")
axs[1,0].legend()
#####################################################################################

#####################################################################################
# Plot simulated and measured C_L/C_D
cl_cd_plot = axs[1,1].plot(angle_smooth, lift_div_drag, label=r"$\frac{C_L}{C_D}$", color="c")
cl_cd_sim_plot = axs[1,1].plot(sim_data[:,0], sim_lift_div_drag, label=r"XFoil $\frac{C_L}{C_D}$", linestyle="--", color="c")
axs[1,1].set_title(r"$\frac{C_L}{C_D}$ vs $\alpha$")
axs[1,1].legend()
#####################################################################################

#####################################################################################
# Set the correct x-axis limits for all plots
for ax in axs.flat:
    ax.set(xlabel=r"$\alpha$", xlim=(-20,20))
#####################################################################################

#####################################################################################
# Give the plots the correct title
fig.suptitle(r"NACA 0018 at $10 \frac{m}{s}$")
plt.show()

#####################################################################################
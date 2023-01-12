import numpy as np
import matplotlib as mpl
#mpl.use('pgf')
import matplotlib.pyplot as plt
import subprocess as sp
import csv
import scipy
import numpy.fft

plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 10
})

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
air_density = 1.293  # Luftdichte
kinematic_viscosity_air = 1.5111E-5 # kinematische Viskosität
flow_speed = 10  # Freistromgeschwindigkeit

angle_filter_length = 1  # Fensterbreite des Moving Median Winkel
force_filter_length = 25 # Fensterbreite des Moving Average Kräfte

axis_diameter = 0.005
axis_length = 0.15

output_directory = "output/"
############################### Constants ###############################

############################### Convert the LabVIEW Waveform Timestamps into a usable Format ###########################
def convert_timestamp(timestamp):
    timestamp = timestamp[9:len(timestamp)].strip()
    hours = int(timestamp[0:2]) * 3600
    minutes = int(timestamp[3:5]) * 60
    seconds = float(timestamp[6:len(timestamp)].replace(",", "."))
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
#####################################################################################

#####################################################################################
# Calculate Reynolds Number
def reynolds(freestream_velocity, characteristic_length, kinematic_viscosity):
    return int((freestream_velocity * characteristic_length) / kinematic_viscosity)
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
angles_inertia = read_angle_data("Real/NACA 0018/dynamic/inertia_signal_3/serial.csv")
angles_inertia = angles_inertia[10000:,:]
angles_inertia[:,0] = np.linspace(np.min(angles_inertia[:,0]), np.max(angles_inertia[:,0]), angles_inertia.shape[0])

forces_inertia = read_force_data("Real/NACA 0018/dynamic/inertia_signal_3/daq.csv")
zero = read_force_data("Real/NACA 0018/dynamic/zero_signal_3/daq.csv")

angles_signal = read_angle_data("Real/NACA 0018/dynamic/signal3_4Hz_30Deg/serial.csv")
angles_signale = angles_signal[10000:,:]
angles_signal[:,0] = np.linspace(np.min(angles_signal[:,0]), np.max(angles_signal[:,0]), angles_signal.shape[0])

forces_signal = read_force_data("Real/NACA 0018/dynamic/signal3_4Hz_30Deg/daq.csv")


#####################################################################################

#####################################################################################
# Merge angle and force data and save them to a file
inertia_merged = sync_merge_data(angles_inertia, forces_inertia)
signal_merged = sync_merge_data(angles_signal, forces_signal)
#####################################################################################

#####################################################################################
re = reynolds(flow_speed, chord_length, kinematic_viscosity_air)
#####################################################################################


#####################################################################################
# Convert all the voltages to forces
f1_inertia = convert_voltage_to_force_vec(inertia_merged[:, 2], f_nom, c1, u_speise, amplifier_sensitivity)
print("f1 converted")

f2_inertia = convert_voltage_to_force_vec(inertia_merged[:, 3], f_nom, c2, u_speise, amplifier_sensitivity)
print("f2 converted")

f3_inertia = convert_voltage_to_force_vec(inertia_merged[:, 4], f_nom, c3, u_speise, amplifier_sensitivity)
print("f3 converted")

f1_signal = convert_voltage_to_force_vec(signal_merged[:, 2], f_nom, c1, u_speise, amplifier_sensitivity)
print("f1 converted")

f2_signal = convert_voltage_to_force_vec(signal_merged[:, 3], f_nom, c2, u_speise, amplifier_sensitivity)
print("f2 converted")

f3_signal = convert_voltage_to_force_vec(signal_merged[:, 4], f_nom, c3, u_speise, amplifier_sensitivity)
print("f3 converted")

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
f1_inertia -= f1_offset
f2_inertia -= f2_offset
f3_inertia -= f3_offset

f1_signal -= f1_offset
f2_signal -= f2_offset
f3_signal -= f3_offset
#####################################################################################

#####################################################################################
# Smooth the angle data using a moving median to remove outliers
angles_inertia_smooth = scipy.ndimage.median_filter(inertia_merged[:, 1], angle_filter_length, mode='reflect')
angles_signal_smooth = scipy.ndimage.median_filter(signal_merged[:, 1], angle_filter_length, mode='reflect')
#####################################################################################

def phase_average(signal, angle):
    zero_crossings = np.where(np.diff(np.sign(angle)))[0]
    zero_crossing_pos = []
    for crossing in zero_crossings:
        if angle[crossing + 1] - angle[crossing - 1] > 0:
            zero_crossing_pos.append(crossing)
    np_crossing_pos = np.array(zero_crossing_pos)
    diff = np.max(np.diff(np_crossing_pos))

    phase_avg = np.zeros(diff)
    num_phases = np.zeros(diff)
    for i in range(0, len(zero_crossing_pos) - 1):
        phase = signal[zero_crossing_pos[i]:zero_crossing_pos[i+1]]

        for j, val in enumerate(phase):
            if num_phases[j] == 0:
                phase_avg[j] = val
            else:
                phase_avg[j] += (val - phase_avg[j]) / num_phases[j]
            num_phases[j] += 1

    return phase_avg

f1_inertia_smooth = np.convolve(f1_inertia, np.ones(force_filter_length) / force_filter_length, mode="same")
f3_inertia_smooth = np.convolve(f3_inertia, np.ones(force_filter_length) / force_filter_length, mode="same")
f3_signal_smooth = np.convolve(f3_signal, np.ones(force_filter_length) / force_filter_length, mode="same")
f1_signal_smooth = np.convolve(f1_signal, np.ones(force_filter_length) / force_filter_length, mode="same")



f1_inertia_avg = phase_average(f1_inertia_smooth, angles_inertia_smooth)
f3_inertia_avg = phase_average(f3_inertia_smooth, angles_inertia_smooth)
f1_signal_avg = phase_average(f1_signal_smooth, angles_inertia_smooth)
f3_signal_avg = phase_average(f3_signal_smooth, angles_inertia_smooth)

f1_corrrected = f1_signal_avg - f1_inertia_avg
f3_corrrected = f3_signal_avg - f3_inertia_avg

lift_coefficient = calculate_lift_coefficient_vec(f1_corrrected, air_density, flow_speed, surface_area)
drag_coefficient = calculate_drag_coefficient_vec(f3_corrrected, air_density, flow_speed, surface_area)

angle_avg = phase_average(signal_merged[:,1], angles_signal_smooth)

width = 6.30045
golden_ratio = (5 ** .5 - 1) / 2
height = width * golden_ratio * 0.8


fig, ax1 = plt.subplots()
fig.set_size_inches(width, height)

ax2 = ax1.twinx()
p1 = ax1.plot(np.arange(lift_coefficient.size)/1.5, lift_coefficient, label="$C_L$")
p2 = ax2.plot(np.arange(angle_avg.size)/1.5, angle_avg, color="grey", label=r"$\alpha$")

ps = p1 + p2
labs = [l.get_label() for l in ps]
ax1.legend(ps, labs, loc=0)

ax1.set_xlabel("$t$")
ax1.set_ylabel("$C_L$")
ax1.set_ylim(-1.5, 1.5)
ax2.set_ylabel(r"$\alpha$")

#plt.savefig('norm.pgf', format='pgf')
fig.tight_layout()
plt.show()
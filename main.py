import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy

def read_angle_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=";")
        for row in reader:
            if row[0] != "":
                row[0] = convert_timestamp(row[0])
                row[1] = float(row[1].replace(",", "."))
                data.append(row)

    print("Read angle data of length")
    print(len(data))
    return np.array(data)


def read_force_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=";")
        for row in reader:
            if row[0] != "":
                row[0] = convert_timestamp(row[0])
                row[1] = float(row[1].replace(",", "."))
                row[2] = float(row[2].replace(",", "."))
                row[3] = float(row[3].replace(",", "."))

                data.append(row)
    print("Read force data of length")
    print(len(data))
    return np.array(data)


def convert_timestamp(timestamp):
    print("Convert timestamp " + timestamp)
    timestamp = timestamp[11:len(timestamp)]
    hours = int(timestamp[0:2]) * 3600
    minutes = int(timestamp[3:5]) * 60
    seconds = float(timestamp[6:len(timestamp)].replace(",", "."))
    print(hours + minutes + seconds)
    return hours + minutes + seconds


def sync_merge_data(angles, forces):
    data = []
    for i, angle in enumerate(angles):
        timestamp_angles = angle[0]
        force1 = np.interp(timestamp_angles, forces[:, 0], forces[:, 1], left=np.nan, right=np.nan)
        force2 = np.interp(timestamp_angles, forces[:, 0], forces[:, 2], left=np.nan, right=np.nan)
        force3 = np.interp(timestamp_angles, forces[:, 0], forces[:, 3], left=np.nan, right=np.nan)
        data.append([timestamp_angles, angle[1], force1, force2, force3])
        print("Synced data point " + str(i))
    return np.array(data)

#angles = read_angle_data("angle.csv")
#forces = read_force_data("forces.csv")

#merged = sync_merge_data(angles, forces)
#np.save("save.npy", merged)

merged = np.load("save.npy", allow_pickle=True)

force_filter_length = 1000
u1_smooth = np.convolve(merged[:, 2], np.ones(force_filter_length) / force_filter_length, mode="same")
u2_smooth = np.convolve(merged[:, 3], np.ones(force_filter_length) / force_filter_length, mode="same")
u3_smooth = np.convolve(merged[:, 4], np.ones(force_filter_length) / force_filter_length, mode="same")



angle_filter_length = 10
angle_smooth = scipy.ndimage.median_filter(merged[:,1], angle_filter_length, mode='reflect')


fig, ax1 = plt.subplots()
ax1.plot(merged[:, 0], u1_smooth, label="$U_1$")
ax1.plot(merged[:, 0], u2_smooth, label="$U_2$")
ax1.plot(merged[:, 0], u3_smooth, label="$U_3$")

ax2 = ax1.twinx()
ax2.plot(merged[:, 0], angle_smooth,  label=r"$\alpha$")

# plt.ylim([-25, 25])
fig.legend()
plt.show()

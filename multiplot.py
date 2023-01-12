import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy

plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 10
})

# m1_angle = np.load("output/messung1_angle.npy")
# m2_angle = np.load("output/messung2_angle.npy")
# m3_angle = np.load("output/messung3_angle.npy")
# m4_angle = np.load("output/messung4_angle.npy")
# m5_angle = np.load("output/messung5_angle.npy")
k5_angle = np.load("output/kalib5_angle.npy")
k6_angle = np.load("output/kalib6_angle.npy")
k7_angle = np.load("output/kalib7_angle.npy")
k8_angle = np.load("output/kalib8_angle.npy")

# m1_lift = np.load("output/messung1_lift.npy")
# m2_lift = np.load("output/messung2_lift.npy")
# m3_lift = np.load("output/messung3_lift.npy")
# m4_lift = np.load("output/messung4_lift.npy")
# m5_lift = np.load("output/messung5_lift.npy")
k5_lift = np.load("output/kalib5_lift.npy")
k6_lift = np.load("output/kalib6_lift.npy")
k7_lift = np.load("output/kalib7_lift.npy")
k8_lift = np.load("output/kalib8_lift.npy")

# m1_drag = np.load("output/messung1_drag.npy")
# m2_drag = np.load("output/messung2_drag.npy")
# m3_drag = np.load("output/messung3_drag.npy")
# m4_drag = np.load("output/messung4_drag.npy")
# m5_drag = np.load("output/messung5_drag.npy")
k5_drag = np.load("output/kalib5_drag.npy")
k6_drag = np.load("output/kalib6_drag.npy")
k7_drag = np.load("output/kalib7_drag.npy")
k8_drag = np.load("output/kalib8_drag.npy")

sim_angle = np.load("output/sim_angle.npy")
sim_lift = np.load("output/sim_lift.npy")
sim_drag = np.load("output/sim_drag.npy")

# plt.plot(m1_angle, m1_lift)
# plt.plot(m2_angle, m2_lift)
# plt.plot(m3_angle, m3_lift)
# plt.plot(m4_angle, m4_lift)
# plt.plot(m5_angle, m5_lift)

k5_angle_interp = np.linspace(np.min(k5_angle), np.max(k5_angle), k5_angle.shape[0])
k8_angle_interp = np.linspace(np.max(k8_angle), np.min(k8_angle), k8_angle.shape[0])

k5_angle_interp = k5_angle_interp[10000:-10000]
k5_lift = k5_lift[10000:-10000]
k5_drag = k5_drag[10000:-10000]
k6_angle = k6_angle[10000:-10000]
k6_lift = k6_lift[10000:-10000]
k6_drag = k6_drag[10000:-10000]
k7_angle = k7_angle[10000:-10000]
k7_lift = k7_lift[10000:-10000]
k7_drag = k7_drag[10000:-10000]
k8_angle_interp = k8_angle_interp[10000:-10000]
k8_lift = k8_lift[10000:-10000]
k8_drag = k8_drag[10000:-10000]


def drag_fit(c, angle, drag):
    range = (-10, 10)
    sum_of_squares = 0
    drag_short = drag[np.where(np.abs(angle) < 2)]
    angle_short = angle[np.where(np.abs(angle) < 2)]
    c_d = drag_short - c
    sim_c_d = np.interp(angle_short, sim_angle, sim_drag)
    square_diff = (c_d - sim_c_d) ** 2
    sum_of_squares = np.sum(square_diff)
    print(c)
    return sum_of_squares


def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


########################################
def resample(signal, sample_rate):
    return np.interp(x=np.arange(sample_rate), xp=np.linspace(0, sample_rate, signal.size), fp=signal, left=np.nan,
                     right=np.nan)


#plt.figure()
#plt.plot(k8_ang, k8_sig)
#plt.plot(k7_ang, k7_sig)
#plt.plot(k6_ang, k6_sig)
#plt.plot(k5_ang, k5_sig)

mins = [np.min(k5_angle_interp), np.min(k6_angle), np.min(k7_angle), np.min(k8_angle_interp)]
maxs = [np.max(k5_angle_interp), np.max(k6_angle), np.max(k7_angle), np.max(k8_angle_interp)]
##################################################
angle_avg = np.linspace(-20, 20, 5000)
k8_s = np.interp(angle_avg, np.flip(k8_angle_interp), np.flip(k8_lift), left=np.nan, right=np.nan)
k7_s = np.interp(angle_avg, k7_angle, k7_lift, left=np.nan, right=np.nan)
k6_s = np.interp(angle_avg, k6_angle, k6_lift, left=np.nan, right=np.nan)
k5_s = np.interp(angle_avg, k5_angle_interp, k5_lift, left=np.nan, right=np.nan)

sigs = np.row_stack((k8_s, k7_s, k6_s, k5_s))

avg_transposition = np.transpose(sigs)

def func_mean(transp):
    means = []
    vari = []
    for row in transp:
        means.append(np.nanmean(row))
        vari.append(np.nanvar(row))
    return np.array(means), np.sqrt(vari)


avg, variance = func_mean(avg_transposition)
v_pos = avg + variance
v_neg = avg - variance
###########################################################
width = 6.30045
golden_ratio = (5 ** .5 - 1) / 2
height = width * golden_ratio * 0.8

plt.figure()
plt.gcf().set_size_inches(width, height)
plt.tight_layout()
plt.fill_between(angle_avg, v_neg, v_pos, alpha=0.2)
plt.plot(angle_avg, avg,  label=r"$\bar{C_L}$")
plt.plot(sim_angle, sim_lift, ls="--", color="grey", label="XFoil $C_L$")

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$C_L$")
plt.xlim(-25, 25)
plt.legend();


#plt.figure(figsize=set_size(455))
#plt.plot(sim_angle, sim_lift, ls="--", color="grey", label="XFoil $C_L$")
#plt.plot(k5_angle_interp, k5_lift, label="$C_{L1}$")
#plt.plot(k6_angle, k6_lift, label="$C_{L2}$")
#plt.plot(k7_angle, k7_lift, label="$C_{L3}$")
#plt.plot(k8_angle_interp, k8_lift, label="$C_{L4}$")
#plt.xlabel(r"$\alpha$")
#plt.ylabel(r"$C_L$")
#plt.xlim(-25, 25)
#plt.legend()

k5_minimum = scipy.optimize.minimize(drag_fit, 0, (k5_angle, k5_drag))
k6_minimum = scipy.optimize.minimize(drag_fit, 0, (k6_angle, k6_drag))
k7_minimum = scipy.optimize.minimize(drag_fit, 0, (k7_angle, k7_drag))
k8_minimum = scipy.optimize.minimize(drag_fit, 0, (k8_angle, k8_drag))
k5_drag -= k5_minimum.x
k6_drag -= k6_minimum.x
k7_drag -= k7_minimum.x
k8_drag -= k8_minimum.x

k8_sd = np.interp(angle_avg, np.flip(k8_angle_interp), np.flip(k8_drag), left=np.nan, right=np.nan)
k7_sd = np.interp(angle_avg, k7_angle, k7_drag, left=np.nan, right=np.nan)
k6_sd = np.interp(angle_avg, k6_angle, k6_drag, left=np.nan, right=np.nan)
k5_sd = np.interp(angle_avg, k5_angle_interp, k5_drag, left=np.nan, right=np.nan)

sigs_d = np.row_stack((k8_sd, k7_sd, k6_sd, k5_sd))

avg_transposition_d = np.transpose(sigs_d)

def func_mean(transp):
    means = []
    vari = []
    for row in transp:
        means.append(np.nanmean(row))
        vari.append(np.nanvar(row))
    return np.array(means), np.sqrt(vari)


avg_d, variance_d = func_mean(avg_transposition_d)
v_pos_d = avg_d + variance_d
v_neg_d = avg_d - variance_d


plt.figure()
plt.gcf().set_size_inches(width, height)
plt.fill_between(angle_avg, v_neg_d, v_pos_d, alpha=0.2)
plt.plot(angle_avg, avg_d,  label=r"$\bar{C_D}$")
plt.plot(sim_angle, sim_drag, ls="--", color="grey", label="XFoil $C_D$")

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\bar{C_D}$")
plt.xlim(-25, 25)
plt.legend()
plt.show()




#plt.figure(figsize=set_size(455))
#plt.plot(sim_angle, sim_drag, ls="--", color="grey", label="XFoil $C_D$")
#plt.plot(k5_angle_interp, k5_drag - k5_minimum.x, label="$C_{D1}$")
#plt.plot(k6_angle, k6_drag - k6_minimum.x, label="$C_{D2}$")
#plt.plot(k7_angle, k7_drag - k7_minimum.x, label="$C_{D3}$")
#plt.plot(k8_angle_interp, k8_drag - k8_minimum.x, label="$C_{D4}$")
#plt.xlabel(r"$\alpha$")
#plt.ylabel(r"$C_D$")
#plt.xlim(-25, 25)
#plt.legend(loc="upper left", framealpha=1, fancybox=False)
#plt.show()

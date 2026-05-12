import numpy as np
import matplotlib.pyplot as plt
import pymaster as nmt
import os
import sys
import tracemalloc
import utils as ut
from scipy.stats import chi2
import mpi_utils as mpi
from itertools import product
from astropy.io import fits


# # General arguments
nsims = 1000
lmax = 1000
lmax_mask = 2*lmax
ls = np.arange(lmax+1)

# NOTE: We are interested in FRB auto-correlations, so these parameters won't
# change.
types = ["cat", "cat", "cat", "cat"]  # Types of the four fields
spins = [0, 0, 0, 0]  # Spins of the four fields
random_seeds = [0, 0, 0, 0]  # Do the four fields share the same mask?

overwrite = False  # Overwrite previously saved products
only_sims = False  # Only compute sim CLs (the heavily parallelized part)
rank, size, comm = mpi.init(switch=True)

# Specific for testing DM mocks
noise_only = True
noise_seed = 102
signal_only = False

# NOTE: Choose a meaningful label to identify the validation case
case = "FRB_noiseonly_seed102"

plot_dir = f"/global/homes/k/kwolz/CatalogCovariancesSandbox/plots/{case}"
out_dir = "/global/homes/k/kwolz/CatalogCovariancesSandbox/out"
os.makedirs(plot_dir, exist_ok=True)
tracemalloc.start()

# Open CHIME data
chime_fn = "/global/homes/k/kwolz/CatalogCovariancesSandbox/data/chime_catalogue_nside4096_noise.fits"
dm = {}
with fits.open(chime_fn) as hdul:
    table_hdu = hdul[1]
    data = hdul[1].data
    ra, dec, dm["noiseless"], dm["gauss"], dm["lognormal"] = [
        np.asarray(data[k], dtype=np.float64)
        for k in ["RA", "DEC", "DM",
                  "DM_with_gaussian_noise", "DM_with_lognormal_noise"]]
    cl_th = np.asarray(hdul[2].data["cell"], dtype=np.float64)

val = dm["gauss"]
# val_noiseless = dm["noiseless"]
msk = np.logical_and(~np.isnan(dec), ~np.isnan(ra), ~np.isnan(val))
dec = dec[msk]
ra = ra[msk]
val = val[msk]
# val_noiseless = val_noiseless[msk]
pos = np.array([np.deg2rad(90.-dec), np.deg2rad(ra)], dtype=np.float64)
num = pos.shape[1]

if rank == 0 and not os.path.isfile(f"{plot_dir}/sigma_high_ell_lmax{lmax}.npy"):
    print("Compute high-ell noise level")
    max_el = 12000
    el = np.arange(max_el+1)
    cl = cl_th[:max_el+1]

    def var_hiell(llo, lhi):
        return np.sum((2*el[llo:lhi+1]+1.)*cl[llo:lhi+1])/(4.*np.pi)

    for lstart in [0, lmax]:
        lprobe = np.arange(lstart, max_el, 10)
        var = np.array([var_hiell(lstart+1, lp) for lp in lprobe])
        plt.plot(lprobe, var, color="r", label=fr"$\ell_={{{lstart}}}0")
    sigma_hiell = np.sqrt(var_hiell(lmax, max_el))
    plt.ylabel(fr"$\sigma^2(\ell_0\leq\ell$")
    plt.xlabel(r"$\ell$")
    plt.savefig(f"{plot_dir}/var_hiell_cl.png")
    plt.close()
    np.save(f"{plot_dir}/sigma_high_ell_lmax{lmax}.npy", np.array([sigma_hiell]))
comm.barrier()

sigma_DM = np.load(f"{plot_dir}/sigma_high_ell_lmax{lmax}.npy")[0]
if signal_only:
    sigma_DM = 0.

cl_th = cl_th[:lmax+1]
mpi.print_rnk0(f"sigma_DM = {sigma_DM}", rank)

mpi.print_rnk0("Generating mock data fields", rank)

if noise_only:
    fac = 0.
alm = [ut.gen_alms(cl_th, 100, spin=sp)*0. for sp in spins]
data = [{"cat": (pos, alm[i], sigma_DM), "seed": 102+random_seeds[i],
         "lmax_mask": lmax_mask}
        for i in range(4)]

# # Getting hash Ids so any two fields with identical hashes are identical.
# NOTE: this is to ensure NaMaster does correctly subtract shot noise.
field_ids = ut.get_field_ids(types, spins, random_seeds)
field_mapping = {id: m for (m, id) in enumerate(field_ids)}
fields_dat_uniq = {id: ut.get_field(lmax, types[m], **(data[m]))
                   for id, m in field_mapping.items()}
fields_dat = [fields_dat_uniq[id] for id in field_ids]

mpi.print_rnk0("Initializing power spectrum calculator", rank)
b = ut.get_bins_from_lmax_log(lmax)
leff = b.get_effective_ells()

if noise_only:
    cl_guess = 4*[None]
else:
    cl_guess = [ut.gen_cl_guess(cl_th, spins[i], spins[j])
                for (i,j) in [(0,2), (0,3), (1,2), (1,3)]]

wsps_dat_uniq = {
    (id, jd): nmt.NmtWorkspace.from_fields(fields_dat[m], fields_dat[n], b)
    for id, m in field_mapping.items() for jd, n in field_mapping.items()
}
wsps_dat = {(m, n): wsps_dat_uniq[id,jd]
            for m, id in enumerate(field_ids)
            for n, jd in enumerate(field_ids)}

f = fields_dat[0]
w = wsps_dat[0, 0]
pcl = nmt.compute_coupled_cell(f, f)
pcl_th = w.couple_cell(np.array([cl_th]))
plt.figure()
plt.plot(ls, pcl_th[0], "k-")
plt.plot(ls, pcl[0], "r--")
plt.loglog()
plt.savefig(f"{plot_dir}/theory_pcl.png")
plt.close()

mpi.print_rnk0(f"Subtract shot noise? {fields_dat[0] is fields_dat[2]}", rank)
cl_i_13 = nmt.get_iNKA_cell(fields_dat[0], fields_dat[2], cl_guess=cl_guess[0])
cl_i_14 = nmt.get_iNKA_cell(fields_dat[0], fields_dat[3], cl_guess=cl_guess[1])
cl_i_23 = nmt.get_iNKA_cell(fields_dat[1], fields_dat[2], cl_guess=cl_guess[2])
cl_i_24 = nmt.get_iNKA_cell(fields_dat[1], fields_dat[3], cl_guess=cl_guess[3])

plt.plot(ls[2:], cl_i_13[0][2:], "g-", label="13")
plt.plot(ls[2:], cl_i_14[0][2:], "r--", label="14")
plt.plot(ls[2:], cl_i_23[0][2:], "b-.", label="23")
plt.plot(ls[2:], cl_i_24[0][2:], "c:", label="24")
plt.legend()
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\bar{C}_\ell$")
plt.loglog()
plt.savefig(f"{plot_dir}/iNKA_cells.png", bbox_inches="tight")
plt.close()

current_gb, peak_gb = [1024**(-3) * c
                       for c in tracemalloc.get_traced_memory()]
print("Memory for field (Current, Peak): "
      f"{current_gb:.2f} GB, {peak_gb:.2f} GB")
tracemalloc.stop()

mpi.print_rnk0(f"Generating {nsims} simulations", rank)
fname = f'{out_dir}/tmp_{case}_cls_nsims{nsims}.npz'
if not os.path.isfile(fname) or overwrite:
    local_sim_ids = mpi.distribute_tasks(size, rank, nsims)
    for i in local_sim_ids:
        print(i)

        # For catalog fields
        r0, r1, r2, r3 = random_seeds
        alm = [ut.gen_alms(cl_th, 1000+i, spin=sp)*fac for sp in spins]
        data = [{"cat": (pos, alm[0], sigma_DM), "seed": 2000*i+r0},
                {"cat": (pos, alm[1], sigma_DM), "seed": 2000*i+r1},
                {"cat": (pos, alm[2], sigma_DM), "seed": 2000*i+r2},
                {"cat": (pos, alm[3], sigma_DM), "seed": 2000*i+r3}]

        # For all fields
        fields_uniq = {id: ut.get_field(lmax, types[m], **(data[m]))
                       for id, m in field_mapping.items()}
        fields = [fields_uniq[id] for id in field_ids]
        print("theta_ipd", fields[0].get_theta_ipd(), "Nf", fields[0].Nf)

        pcl = {(m, n): nmt.compute_coupled_cell(f1, f2)
               for m, f2 in enumerate(fields)
               for n, f1 in enumerate(fields)}
        np.savez(f"{out_dir}/tmp_{case}_pcl_sim{i:04}", pcl=pcl)
        if "mom" in types or "num" in types:
            wsps_uniq = {
                (id, jd): nmt.NmtWorkspace.from_fields(fields[m], fields[n], b)
                for id, m in field_mapping.items()
                for jd, n in field_mapping.items()
            }
            w = {(m, n): wsps_uniq[id,jd]
                 for m, id in enumerate(field_ids)
                 for n, jd in enumerate(field_ids)}
        else:
            w = wsps_dat
        cl = {(m, n): w[m,n].decouple_cell(pcl[m,n])
              for m in range(4) for n in range(4)}
        np.savez(f"{out_dir}/tmp_{case}_cl_sim{i:04}.npz", cl=cl)
    comm.barrier()
    if rank == 0:
        pcls = {(i, j): [] for i in range(4) for j in range(4)}
        cls = {(i, j): [] for i in range(4) for j in range(4)}
        
        for i in range(nsims):
            (pcl_fn, cl_fn) = (f"{out_dir}/tmp_{case}_{lab}_sim{i:04}.npz"
                                   for lab in ["pcl", "cl"])
            pcli = np.load(pcl_fn, allow_pickle=True)["pcl"].item()
            cli = np.load(cl_fn, allow_pickle=True)["cl"].item()
            for i, j in product([0, 1, 2, 3], [0, 1, 2, 3]):
                pcls[i, j].append(pcli[i,j])
                cls[i, j].append(cli[i,j])
            os.remove(pcl_fn)
            os.remove(cl_fn)
        
        for i, j in product([0, 1, 2, 3], [0, 1, 2, 3]):
            pcls[i,j] = np.array(pcls[i,j])
            cls[i,j] = np.array(cls[i,j])
        np.savez(fname, cls=cls, pcls=pcls)
elif rank == 0:
    d = np.load(fname, allow_pickle=True)
    pcls = d["pcls"].item()
    cls = d["cls"].item()

if only_sims or rank != 0:
    sys.exit()

if not os.path.isfile(f"{out_dir}/cov_{case}_pcl_ana.npy") or overwrite:
    print("Computing sim-based covariance")

    if not noise_only:
        # Plot sims vs. theory
        fig, axes = plt.subplots(4, 4, figsize=(15, 15),
            gridspec_kw={"hspace": 0.4, "wspace": 0.3}, sharex=True
        )
        for isp, (is1, is2) in enumerate([[0,2], [0,3], [1,2], [1,3]]):
            ncl = cl_guess[isp].shape[0]
            for icl in range(ncl):
                plab = {1: ["TT"], 2: ["TE", "TB"], 4: ["EE", "EB", "BE", "BB"]}[ncl][icl]
                ax = axes[isp, icl]
                ax.plot(ls[2:], np.mean(pcls[is1,is2][:, icl, 2:], axis=0), "r-", label=f"{is1+1}{is2+1}, {plab}")
                ax.plot(ls[2:], wsps_dat[is1,is2].couple_cell(cl_guess[isp])[icl, 2:], "k-.", label="Theory", zorder=32)
                ax.set_yscale('log')
                ax.set_xlabel(r"$\ell$")
                ax.set_ylabel(fr"$pC_\ell^{{{plab}}}$")
                ax.axvspan(lmax, lmax, alpha=0.5, color='gray')
                ax.legend()
        plt.savefig(f"{plot_dir}/sims_cells.png", bbox_inches="tight")
        plt.close()

    print("cl")
    tracemalloc.start()
    cov_cl = (np.mean(cls[0,1][:, None, :, None, :]*cls[2,3][:, :, None, :, None], axis=0) -
            np.mean(cls[0,1][:, None, :, None, :], axis=0) *
            np.mean(cls[2,3][:, :, None, :, None], axis=0))
    print("pcl")
    cov_pcl = (np.mean(pcls[0,1][:, None, :, None, :]*pcls[2,3][:, :, None, :, None], axis=0) -
            np.mean(pcls[0,1][:, None, :, None, :], axis=0) *
            np.mean(pcls[2,3][:, :, None, :, None], axis=0))

    print("Computing analytical covariance")
    cw = nmt.NmtCovarianceWorkspace.from_fields(*fields_dat)

    print("pcl")
    n12, n34 = [ut.num_spin_comp(spins[0], spins[1]),
                ut.num_spin_comp(spins[2], spins[3])]
    cov_pcl_ana = cw.gaussian_covariance(
        cl_i_13, cl_i_14, cl_i_23, cl_i_24, wa=wsps_dat[0,1], wb=wsps_dat[2,3],
        coupled=True
    ).reshape((len(ls), n12, len(ls), n34))
    print("cl")
    cov_cl_ana = cw.gaussian_covariance(
        cl_i_13, cl_i_14, cl_i_23, cl_i_24, wa=wsps_dat[0,1], wb=wsps_dat[2,3]
    ).reshape((len(leff), n12, len(leff), n34))

    np.save(f"{out_dir}/cov_{case}_pcl_ana.npy", cov_pcl_ana)
    np.save(f"{out_dir}/cov_{case}_pcl_sim.npy", cov_pcl)
    np.save(f"{out_dir}/cov_{case}_cl_ana.npy", cov_cl_ana)
    np.save(f"{out_dir}/cov_{case}_cl_sim.npy", cov_cl)
cov_pcl_ana = np.load(f"{out_dir}/cov_{case}_pcl_ana.npy")
cov_pcl = np.load(f"{out_dir}/cov_{case}_pcl_sim.npy")
cov_cl_ana = np.load(f"{out_dir}/cov_{case}_cl_ana.npy")
cov_cl = np.load(f"{out_dir}/cov_{case}_cl_sim.npy")

pol_pairs = {(0, 0): ["TT"], (0, 2): ["TE", "TB"], (2, 0): ["ET", "EB"],
             (2, 2): ["EE", "EB", "BE", "BB"]}
pol_pairs = ["XY"] if spins[:2] != spins[2:] else pol_pairs[spins[0], spins[1]]

for ip, pol_pair in enumerate(pol_pairs):
    plt.figure()
    plt.plot(ls[2:], np.diag(cov_pcl_ana[2:, ip, 2:, ip]), 'r-', label='Analytic')
    plt.plot(ls[2:], np.diag(cov_pcl[ip, ip, 2:, 2:]), 'k:', alpha=0.5, label='Sims')
    plt.loglog()
    plt.xlabel(r'$\ell$', fontsize=18)
    plt.ylabel(r'${\rm Cov}(pC_\ell^{ab},pC_{\ell}^{cd})$', fontsize=18)
    plt.legend(fontsize=14, frameon=False, ncol=2)
    plt.savefig(f"{plot_dir}/cov_pcl_{pol_pair}.png", bbox_inches="tight")
    plt.close()

for ip, pol_pair in enumerate(pol_pairs):
    plt.figure()
    plt.plot(leff, np.diag(cov_cl_ana[:, ip, :, ip]), 'r-', label='Analytic')
    plt.plot(leff, np.diag(cov_cl[ip, ip, :, :]), 'k:', alpha=0.5, label='Sims')
    plt.loglog()
    plt.xlabel(r'$\ell$', fontsize=18)
    plt.ylabel(r'${\rm Cov}(C_\ell^{ab},C_{\ell}^{cd})$', fontsize=18)
    plt.legend(fontsize=14, frameon=False, ncol=2)
    plt.savefig(f"{plot_dir}/cov_cl_{pol_pair}.png", bbox_inches="tight")
    plt.close()

current_gb, peak_gb = [1024**(-3) * c
                       for c in tracemalloc.get_traced_memory()]
print("memory for covariance (Current, Peak): "
      f"{current_gb:.2f} GB, {peak_gb:.2f} GB")
tracemalloc.stop()

print("Doing chi2 tests")
r0, r1, r2, r3 = random_seeds
f0, f1, f2, f3 = field_ids
s0, s1, s2, s3 = spins
t0, t1, t2, t3 = types
for ip, pol_pair in enumerate(pol_pairs):
    p0, p1 = pol_pair
    for lmax_plot in [lmax]:
        lmsk = leff <= lmax_plot
        dxa = cls[r0,r1][:, ip, lmsk] - np.mean(cls[r0,r1][:, ip, lmsk], axis=0)
        dxb = cls[r2,r3][:, ip, lmsk] - np.mean(cls[r2,r3][:, ip, lmsk], axis=0)
        invcov = np.linalg.inv(cov_cl_ana[:, ip, :, ip])
        invcov_sim = np.linalg.inv(cov_cl[ip, ip, :, :])

        chi2_ana = np.einsum("mn, nm->m", dxa, np.dot(invcov[lmsk, :][:, lmsk], dxb.T))
        chi2_sim = np.einsum("mn, nm->m", dxa, np.dot(invcov_sim[lmsk, :][:, lmsk], dxb.T))

        ndof = np.sum(lmsk)
        x = np.arange(100)
        chi2_pdf = chi2.pdf(x, ndof)
        title = fr"Cov(${p0}_{{{f0}}}{p1}_{{{f1}}},\,{p0}_{{{f2}}}{p1}_{{{f3}}}$) " +r"$\ell_{\rm max}=$"+str(lmax_plot)
        plt.figure()
        plt.title(title)
        plt.hist(chi2_ana, label="iNKA", bins=30, lw=2, density=True, histtype="step", alpha=0.7)
        plt.hist(chi2_sim, label="Gaussian sims", bins=30, lw=2, density=True, histtype="step", alpha=0.7)
        if (t0, t1) == (t2, t3) and (s0, s1) == (s2, s3):
            plt.plot(x, chi2_pdf, c="k", ls="--", label=fr"$\chi^2$ dist. $(N_{{\rm dof}}={{{ndof}}})$")
        plt.xlabel(r"$\chi^2$")
        plt.ylabel(r"$p(\chi^2)$")
        plt.legend()
        plt.savefig(f"{plot_dir}/chi2_{pol_pair}_lmax{lmax_plot}.png")
        plt.close()

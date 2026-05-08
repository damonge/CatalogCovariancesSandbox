import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pymaster as nmt
import os
import sys
import wget
import tracemalloc
import utils as ut
import mpi_utils as mpi

# # General arguments
nsims = 1000
# NOTE: Choose a meaningful label to identify the validation case
case = "momentum_spin2_randoms_allsame"
# plot_dir = f"/shared_home/kwolz/bbdev/catalog_pcl/plots/{case}"
plot_dir = f"/global/homes/k/kwolz/CatalogCovariancesSandbox/plots/{case}"
out_dir = "/global/homes/k/kwolz/CatalogCovariancesSandbox/out"
# out_dir = "/shared_home/kwolz/bbdev/catalog_pcl/out"
os.makedirs(plot_dir, exist_ok=True)
tracemalloc.start()

nside = 256
fname_mask = 'selection_function_NSIDE64_G20.5_zsplit2bin0.fits'  # noqa
if not os.path.isfile(fname_mask):
    wget.download("https://zenodo.org/records/8098636/files/selection_function_NSIDE64_G20.5_zsplit2bin0.fits?download=1")  # noqa
msk = hp.ud_grade(hp.read_map(fname_mask), nside_out=nside)
msk /= np.amax(msk)
# msk = np.ones(hp.nside2npix(nside))

nhope = 100000
nran_factor = 20
lmax = 3*nside-1
ls = np.arange(lmax+1)
types = ["mom", "mom", "mom", "mom"]  # Types of the four fields
spins = [2, 2, 2, 2]  # Spins of the four fields
random_seeds = [0, 0, 0, 0]  # Do the four fields share the same mask?

overwrite = False  # Overwrite previously saved products
only_sims = False  # Only compute sim CLs (the heavily parallelized part)
rank, size, comm = mpi.init(switch=True)

mpi.print_rnk0("Generating mock data fields", rank)
# # NOTE: Comment the irrelevant cases below

# # For catalog fields
# pos = [ut.get_catalog(nhope, msk, 1000+r) for r in random_seeds]
# cl_th = 1./(10+ls)**0.7
# # Accounting for artificial pixel window from HEALPix pixels
# # cl_th *= hp.pixwin(nside, lmax=lmax)
# alm = [ut.gen_alms(cl_th, 100, spin=sp) for sp in spins]
# data = [{"cat": (pos[0], alm[0])},
#          {"cat": (pos[1], alm[1])},
#          {"cat": (pos[2], alm[2])},
#          {"cat": (pos[3], alm[3])}]

# # For map fields
# cl_th = 1./(10+ls)
# pixwin = None
# map = [ut.gen_maps(cl_th, 100, nside, spin=sp) for sp in spins]
# data = [{"map": map[0], "msk": msk},
#         {"map": map[1], "msk": msk},
#         {"map": map[2], "msk": msk},
#         {"map": map[3], "msk": msk}]

# For clustering or momentum fields
pixwin = hp.pixwin(nside, lmax=lmax)
cl_th, cl_od, cl_v, cl_th_pw, cl_od_pw = ut.get_momentum_cl(
    lmax, plot_dir, pl_index=1.2, std_offset=5, pl_index_v=2.2,
    is_clustering=("mom" not in types), overwrite=True, pixwin=pixwin
)
alm_od = hp.synalm(cl_od)
vlm = [ut.gen_alms(cl_v, 100, spin=sp) for sp in spins]
odmap = hp.alm2map(alm_od, nside)
posNum = [ut.get_catalog(nhope, (1 + odmap)*msk, 100+r) for r in random_seeds]
posRan = [ut.get_catalog(nhope*nran_factor, msk, 200+r) for r in random_seeds]
data = [{"cat": (posNum[0], vlm[0]), "msk": None, "ran": posRan[0]},
        {"cat": (posNum[1], vlm[1]), "msk": None, "ran": posRan[1]},
        {"cat": (posNum[2], vlm[2]), "msk": None, "ran": posRan[2]},
        {"cat": (posNum[3], vlm[3]), "msk": None, "ran": posRan[3]}]

# Getting hash Ids so any two fields with identical hashes are identical.
field_ids = ut.get_field_ids(types, spins, random_seeds)
fields_dat_uniq = [ut.get_field(lmax, types[i], **(data[i]))
                   for i in list(set(field_ids))]
fields_dat = [fields_dat_uniq[i] for i in field_ids]

mpi.print_rnk0("Initializing power spectrum calculator", rank)
b = ut.get_bins_from_lmax_log(lmax)
leff = b.get_effective_ells()
cl_guess = [ut.gen_cl_guess(cl_th_pw, spins[i], spins[j])
            for (i, j) in [(0,2), (0,3), (1,2), (1,3)]]
wsps_dat_uniq = [[nmt.NmtWorkspace.from_fields(fields_dat[i], fields_dat[j], b)
             for j in list(set(field_ids))] for i in list(set(field_ids))]
wsps_dat = [[wsps_dat_uniq[i][j] for j in field_ids] for i in field_ids]

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
        # # NOTE: Comment the irrelevant cases in the same way as above.

        # # For catalog fields
        # alm = [ut.gen_alms(cl_th, 1000+i, spin=sp) for sp in spins]
        # data = [{"cat": (pos[0], alm[0])},
        #         {"cat": (pos[1], alm[1])},
        #         {"cat": (pos[2], alm[2])},
        #         {"cat": (pos[3], alm[3])}]

        # # For map fields
        # map = [ut.gen_maps(cl_th, 1000+i, nside, spin=sp) for sp in spins]
        # data = [{"map": map[0], "msk": msk},
        #         {"map": map[1], "msk": msk},
        #         {"map": map[2], "msk": msk},
        #         {"map": map[3], "msk": msk}]

        # For clustering or momentum fields
        alm_od = ut.gen_alms(cl_od, 1000+i)
        vlm = [ut.gen_alms(cl_v, 2000+i, spin=sp) for sp in spins]
        odmap = hp.alm2map(alm_od, nside)
        posNum = [ut.get_catalog(nhope, (1 + odmap)*msk, 1000*i+r) for r in random_seeds]
        posRan = [ut.get_catalog(nhope*nran_factor, msk, 2000*i+r) for r in random_seeds]
        data = [{"cat": (posNum[0], vlm[0]), "msk": None, "ran": posRan[0]},
                {"cat": (posNum[1], vlm[1]), "msk": None, "ran": posRan[1]},
                {"cat": (posNum[2], vlm[2]), "msk": None, "ran": posRan[2]},
                {"cat": (posNum[3], vlm[3]), "msk": None, "ran": posRan[3]}]

        # For all fields
        fields_uniq = [ut.get_field(lmax, types[i], **(data[i]))
                       for i in list(set(field_ids))]
        fields = [fields_uniq[i] for i in field_ids]
        pcl = np.array([[nmt.compute_coupled_cell(f1, f2)
                        for f2 in fields]
                        for f1 in fields])
        np.save(f"{out_dir}/tmp_{case}_pcl_sim{i:04}", pcl)
        if "mom" in types or "num" in types:
            wsps_uniq = [[nmt.NmtWorkspace.from_fields(fields[i], fields[j], b)
                        for j in list(set(field_ids))]
                        for i in list(set(field_ids))]
            w = [[wsps_uniq[field_ids[i]][field_ids[j]]
                  for j in range(4)] for i in range(4)]
        else:
            w = wsps_dat
        cl = np.array([[w[i][j].decouple_cell(pcl[i, j])
                      for j in range(4)] for i in range(4)])
        np.save(f"{out_dir}/tmp_{case}_cl_sim{i:04}", cl)
    comm.barrier()
    if rank == 0:
        pcls = []
        cls = []
        for i in range(nsims):
            (pcl_fn, cl_fn) = (f"{out_dir}/tmp_{case}_{lab}_sim{i:04}.npy"
                                for lab in ["pcl", "cl"])
            pcls.append(np.load(pcl_fn))
            cls.append(np.load(cl_fn))
            os.remove(pcl_fn)
            os.remove(cl_fn)
        pcls = np.array(pcls)
        cls = np.array(cls)
        np.savez(fname, cls=cls, pcls=pcls)
elif rank == 0:
    d = np.load(fname)
    pcls = d["pcls"]
    cls = d["cls"]

if only_sims or rank != 0:
    sys.exit()

print("Computing sim-based covariance")
plt.figure()
plt.plot(ls[2:], wsps_dat[0][2].couple_cell(cl_guess[0])[0, 2:], "k-", label="Theory")
plt.plot(ls[2:], np.mean(pcls[:, 0, 2, 0, 2:], axis=0), "r-.", label="Sims")
plt.yscale('log')
plt.xlabel(r"$\ell$")
plt.ylabel(r"$pC_\ell$")
plt.axvline(2*256, color="k")  # DEBUG
plt.axvspan(2*nside, lmax, alpha=0.5, color='gray')
plt.legend()
plt.savefig(f"{plot_dir}/sims_cells.png")

# Note that here, we are ignoring the cross-covariance between different
# mode pairs (e.g. EE x EB) but only auto covariances (e.g. EB x EB). We could
# generalize this but it would be more expensive.
print("cl")
tracemalloc.start()
cov_cl = (np.mean(cls[:, 0, 1, :, None, :]*cls[:, 2, 3, :, :, None], axis=0) -
          np.mean(cls[:, 0, 1, :, None, :], axis=0) *
          np.mean(cls[:, 2, 3, :, :, None], axis=0))
print("pcl")
cov_pcl = (np.mean(pcls[:, 0, 1, :, None, :]*pcls[:, 2, 3, :, :, None], axis=0) -
           np.mean(pcls[:, 0, 1, :, None, :], axis=0) *
           np.mean(pcls[:, 2, 3, :, :, None], axis=0))

print("Computing analytical covariance")
cw = nmt.NmtCovarianceWorkspace.from_fields(*fields_dat)

print("pcl")
n12, n34 = [ut.num_spin_comp(spins[0], spins[1]),
            ut.num_spin_comp(spins[2], spins[3])]
cov_pcl_ana = cw.gaussian_covariance(
    cl_i_13, cl_i_14, cl_i_23, cl_i_24, wa=wsps_dat[0][1], wb=wsps_dat[2][3],
    coupled=True
).reshape((len(ls), n12, len(ls), n34))
print("cl")
cov_cl_ana = cw.gaussian_covariance(
    cl_i_13, cl_i_14, cl_i_23, cl_i_24, wa=wsps_dat[0][1], wb=wsps_dat[2][3]
).reshape((len(leff), n12, len(leff), n34))

np.save(f"{out_dir}/cov_{case}_pcl_ana.npy", cov_pcl_ana)
np.save(f"{out_dir}/cov_{case}_pcl_sim.npy", cov_pcl)
np.save(f"{out_dir}/cov_{case}_cl_ana.npy", cov_cl_ana)
np.save(f"{out_dir}/cov_{case}_cl_sim.npy", cov_cl)

pol_pairs = {(0, 0): ["TT"], (0, 2): ["TE", "TB"], (2, 0): ["ET", "EB"],
             (2, 2): ["EE", "EB", "BE", "BB"]}
pol_pairs = ["cross"] if spins[:2] != spins[2:] else pol_pairs[spins[0], spins[1]]

for ip, pol_pair in enumerate(pol_pairs):
    plt.figure()
    plt.plot(ls[2:], np.diag(cov_pcl_ana[2:, ip, 2:, ip]), 'r-', label='Analytic')
    plt.plot(ls[2:], np.diag(cov_pcl[ip, 2:, 2:]), 'k:', alpha=0.5, label='Sims')
    plt.loglog()
    plt.axvline(2*256, color="k")  # DEBUG
    plt.xlabel(r'$\ell$', fontsize=18)
    plt.ylabel(r'${\rm Cov}(pC_\ell^{ab},pC_{\ell}^{cd})$', fontsize=18)
    plt.legend(fontsize=14, frameon=False, ncol=2)
    plt.savefig(f"{plot_dir}/cov_pcl_{pol_pair}.png", bbox_inches="tight")
    plt.close()

for ip, pol_pair in enumerate(pol_pairs):
    plt.figure()
    plt.plot(leff, np.diag(cov_cl_ana[:, ip, :, ip]), 'r-', label='Analytic')
    plt.plot(leff, np.diag(cov_cl[ip, :, :]), 'k:', alpha=0.5, label='Sims')
    plt.loglog()
    plt.axvline(2*256, color="k")  # DEBUG
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

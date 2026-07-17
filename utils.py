import numpy as np
import healpy as hp
import pymaster as nmt
import os
import wget
from collections import OrderedDict
from astropy.io import fits
import scipy.stats as stats
import matplotlib.pyplot as plt


def unique_non_nan_column_indices(a, return_complement=False):
    """
    Returns indices of a mxn-dimensional array such that the resulting array:
    - does not contain any NaN values in any column
    - does not contain any duplicate columns (it selects first occurence)
    - preserves the original ordering
    """
    valid = ~np.isnan(a).any(axis=0)
    _, idx = np.unique(a[:, valid], axis=1, return_index=True)
    col_indices = np.where(valid)[0][idx]
    keep = np.sort(col_indices)

    if return_complement:
        all_cols = np.arange(a.shape[1])
        return np.setdiff1d(all_cols, keep)

    return keep


def load_CHIME_catalog(lmax):
    """
    Returns CHIME catalog positions, theory Cls (up to lmax),
    and dispersion measure catalogs noiseless, with Gaussian noise,
    and with lognormal noise, respectively.
    """
    chime_fn = "/global/homes/k/kwolz/CatalogCovariancesSandbox/data/chime_catalogue_nside4096_noise.fits"  # noqa: E501
    with fits.open(chime_fn) as hdul:
        data = hdul[1].data
        ra, dec, dm, dm_gauss, dm_lognorm = [
            np.asarray(data[k], dtype=np.float64)
            for k in ["RA", "DEC", "DM", "DM_with_gaussian_noise",
                      "DM_with_lognormal_noise"]]
        cl_th = np.asarray(hdul[2].data["cell"], dtype=np.float64)[:lmax+1]
    pos = np.array([np.deg2rad(90.-dec), np.deg2rad(ra)], dtype=np.float64)
    msk = np.logical_and(~np.isnan(dec), ~np.isnan(ra), ~np.isnan(dm))
    pos, dm, dm_gauss, dm_lognorm = pos[:, msk], dm[msk], dm_gauss[msk], dm_lognorm[msk]
    pos, idx = np.unique(pos, axis=1, return_index=True)

    return cl_th, pos, dm[idx], dm_gauss[idx], dm_lognorm[idx]


def load_DES_catalog():
    """
    Returns positions, weights, and shear field values of the DES Y3 catalog
    (maglim shear product) at redshift bin 3.
    """
    import sys
    sys.path.insert(0, "/global/homes/k/kwolz/software/Cosmotheka")
    from cosmotheka.mappers.mapper_DESY3wl import MapperDESY3wl
    print("Loading DES catalog")
    cat_fn = '/pscratch/sd/k/kwolz/Datasets/DES_Y3/catalogs/DESY3_indexcat_zbin3.h5'
    if not os.path.isfile(cat_fn):
        wget.download("http://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/y3kp_cats/DESY3_indexcat.h5", out=cat_fn)  # noqa
    # prod_fn = "/pscratch/sd/k/kwolz/Datasets/DES_Y3-data_products/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits"
    # if not os.path.isfile(prod_fn):
    #     wget.download("http://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/datavectors/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits", out=prod_fn)  # noqa
    config = {
        "zbin": 3,
        "mode": "shear",
        "indexcat": '/mnt/extraspace/damonge/Datasets/DES_Y3/catalogs/DESY3_indexcat_zbin3.h5',
        'file_nz': '/mnt/extraspace/damonge/Datasets/DES_Y3/data_products/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits',
        'nside': 4096,
        'coords': 'C',
        'bin_edges': [0, 30, 60, 90, 120, 150, 180, 210, 240, 272, 309, 351, 398, 452, 513, 582, 661, 750, 852, 967, 1098, 1247, 1416, 1608, 1826, 2073, 2354, 2673, 3035, 3446, 3914, 4444, 5047, 5731, 6508, 7390, 8392, 9529, 10821, 12288, 13947, 15830, 17967, 20393, 24576],
    }
    m = MapperDESY3wl(config)
    weights = m.get_weights()
    ra = m.get_positions()['ra']
    dec = m.get_positions()['dec']
    e1, e2 = m.get_ellips_unbiased()
    pos = np.radians(np.array([90. - dec, ra])).astype(np.float64)
    shear = np.array([-e1, e2]).astype(np.float64)

    return pos, weights, shear


def load_theory_cls(lmax, fname, ncol=2, has_pixwin=False, nside=None):
    ls = np.loadtxt(fname, usecols=0)
    cls = []
    for icol in range(1, ncol):
        cls_append = np.loadtxt(fname, dtype=np.float64, usecols=icol)
        cls.append(np.concatenate(([0.]*int(ls[0]), cls_append)))
    cls = np.array(cls).flatten()[:lmax+1] if ncol == 2 else np.array(cls)[:lmax+1]

    if has_pixwin:
        if not nside:
            raise ValueError("To correct for pixwin, "
                             "you need to provide nside.")
        bl = hp.pixwin(nside, lmax=lmax)
        cls *= bl**2
    return cls


def get_catalog(npoints, m, seed, verbose=False):
    """
    Generates point catalog given a modulating mask m.

    Parameters:
        npoints: upper limit to number of points generated
        m: spin-0 map. modulation mask
        seed: random seed
    Returns:
        pos: Catalog positions
    """
    np.random.seed(seed)
    m = m / np.amax(m)  # new: normalize to [0,1]
    npix = len(m)
    npoints_get = int(npoints/np.mean(m))
    if npoints_get > 1e7 and verbose:
        print(f"WARNING: Npoints = {npoints_get}. "
              "Consider decreasing nhope or steepening power spectrum slope.")
    phi = 2*np.pi*np.random.rand(npoints_get)
    th = np.arccos(-1+2*np.random.rand(npoints_get))
    ipix = hp.ang2pix(hp.npix2nside(npix), th, phi)
    mv = m[ipix]
    u = np.random.rand(npoints_get)
    keep = u <= mv
    th = th[keep]
    phi = phi[keep]
    return np.array([th, phi], dtype=np.float64)


def gen_random(nsrc, mask):
    """
    """
    pmap = mask / np.amax(mask)
    nsrc_hi = int(nsrc/np.mean(pmap))  # not sure what this does
    nside = hp.npix2nside(len(pmap))
    cth = -1 + 2*np.random.rand(int(nsrc_hi))
    phi = 2*np.pi*np.random.rand(int(nsrc_hi))
    sth = np.sqrt(1 - cth**2)
    unif = np.random.rand(nsrc_hi)
    vec = np.array([sth*np.cos(phi), sth*np.sin(phi), cth])
    ipix = hp.vec2pix(nside, *vec)
    good = unif <= pmap[ipix]
    positions = np.array([np.arccos(cth[good]), phi[good]])
    return positions, ipix[good]


def gen_alms(cl, seed, spin=0, has_b=True, TB=False):
    """
    """
    if cl is None:
        return None
    np.random.seed(seed)
    alm1 = hp.synalm(cl)
    if spin == 0:
        return alm1
    np.random.seed(seed+12345)  # TE nonzero but TB zero
    alm2 = hp.synalm(cl)
    if not has_b:
        alm2 *= 0.
    if not TB:  # no TB correlation
        return np.array([alm1, alm2])
    return np.array([alm2, alm1])


def gen_maps(cl, seed, nside, spin=0):
    """
    """
    np.random.seed(seed)
    if spin == 0:
        return hp.synfast(cl, nside)
    map1 = hp.synfast(cl, nside)
    np.random.seed(seed+12345)
    map2 = hp.synfast(cl, nside)
    return np.array([map1, map2])


def alm2map(alm, nside):
    """
    """
    alm = np.array(alm)
    if np.atleast_2d(alm).shape[0] == 1:
        return hp.alm2map(alm, nside)
    elif alm.ndim == 2 and alm.shape[0] == 2:
        return hp.alm2map_spin(alm, nside, 2)
    else:
        raise ValueError(f"alm has wrong input shapr ({alm.shape})")


def gen_cl_guess(cl0, spin1=0, spin2=0, has_b=True):
    """
    """
    if (spin1, spin2) == (0, 0):
        return np.array([cl0])
    elif spin1 == 0 or spin2 == 0:
        return np.array([cl0, 0*cl0])  # TE nonzero but TB zero
    else:
        if has_b:
            return np.array([cl0, 0*cl0, 0*cl0, cl0])
        return np.array([cl0, 0*cl0, 0*cl0, 0*cl0])


def num_spin_comp(spin1, spin2):
    """
    Returns the number of spin components of a CL given spin pair.
    """
    if (spin1, spin2) == (0, 0):
        return 1
    elif spin1 == 0 or spin2 == 0:
        return 2
    return 4


def _evaluate_product_cl(lmax, cls_cc, cls_bb, plot_dir, overwrite=True,
                         lab=None):
    """
    Evaluates power spectrum of field a = b*c, where b and c are uncorrelated
    Gaussian fields with powerspectra cls_cc and cls_bb, as

    .. math::
      \\C^{aa}_\\ell = \\sum_{\\ell_1\\ell_2}
      \\frac{(2\\ell_1+1)(2\\ell_2+1)}{4\\pi}
      \\C^{bb}_\\ell_1
      \\left(\\begin{array}{ccc}
      \\ell & \\ell_1 & \\ell_2 \\\\
      0 & 0 & 0
      \\end{array}\\right)^2
      \\C^{cc}_\\ell_2
    """
    label = "" if lab is None else lab
    if os.path.isfile(f"{plot_dir}/theory_spectrum_{label}.npy") and not overwrite:
        return np.load(f"{plot_dir}/theory_spectrum_{label}.npy")

    from pyshtools.utils import Wigner3j  # noqa
    ells_w3j = np.arange(lmax+1)
    w3j = np.zeros_like(ells_w3j, dtype=float)
    big_w3j = np.zeros((lmax+1, lmax+1, lmax+1))
    for ell1 in ells_w3j[1:]:
        # if ell1 % 100 == 0:
        #     print(" ", ell1)
        for ell2 in ells_w3j[1:]:
            w3j_array, ellmin, ellmax = Wigner3j(ell1, ell2, 0, 0, 0)
            w3j_array = w3j_array[:ellmax - ellmin + 1]
            # make the w3j_array the same shape as the w3j
            if len(w3j_array) < len(ells_w3j):
                reference = np.zeros(len(w3j))
                reference[:w3j_array.shape[0]] = w3j_array
                w3j_array = reference

            w3j_array = np.concatenate([w3j_array[-ellmin:],
                                        w3j_array[:-ellmin]])
            w3j_array = w3j_array[:len(ells_w3j)]
            w3j_array[:ellmin] = 0

            big_w3j[:, ell1, ell2] = w3j_array

    big_w3j = big_w3j**2

    ls = np.arange(lmax+1)
    v_left = (2*ls+1) * cls_cc[:lmax+1]
    v_right = (2*ls+1) * cls_bb[:lmax+1]

    mat = big_w3j
    product_cl = np.dot(np.dot(mat, v_right), v_left) / (4*np.pi)
    np.save(f"{plot_dir}/theory_spectrum", product_cl)

    return product_cl


def check_gaussian(x, plot_fname=None):
    """
    Applies a series of Gaussianity checks to a given set of test values x
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]

    mu = np.mean(x)
    sigma = np.std(x, ddof=1)

    print(f"mean = {mu:.4g}")
    print(f"std  = {sigma:.4g}")
    print(f"skewness = {stats.skew(x):.4g}")
    print(f"excess kurtosis = {stats.kurtosis(x):.4g}")

    # Shapiro-Wilk is good for small-to-medium samples
    if len(x) <= 5000:
        _, p = stats.shapiro(x)
        print(f"Shapiro-Wilk p-value = {p:.4g}")
    else:
        _, p = stats.normaltest(x)
        print(f"D'Agostino-Pearson p-value = {p:.4g}")

    # Q-Q plot
    plt.figure()
    stats.probplot(x, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot")
    plt.savefig(plot_fname)
    plt.close()


def get_pte(chi2_in, chi2_dof=None, chi2_sims=None):
    """
    Given either a theoretical chi2 distribution or a set of simulated values,
    compute the probability to exceed.
    """
    from scipy.stats import rv_histogram, chi2
    if chi2_dof is not None:
        return chi2.sf(chi2_in, chi2_dof)
    if chi2_sims is not None:
        counts, bin_edges = np.histogram(
            chi2_sims,
            bins=min(30, len(chi2_sims)//10), density=True)
        histogram_dist = rv_histogram((counts, bin_edges))

        return histogram_dist.sf(chi2_in)


def uniformity_test(unif_samples, thresh=0.05):
    """
    Defines a passing criterion for a set of test samples. The test passes
    if the KS test against uniformity in [0, 1) yields a p-value outside
    (5%, 95%).
    """
    from scipy.stats import kstest
    _, pval = kstest(unif_samples, "uniform", args=(0, 1))
    return np.logical_and(pval > thresh, 1-thresh > pval), pval


def smooth_broken_power_law(x, A=1.0, xb=1.0, alpha1=1.0, alpha2=2.0, s=1.0):
    """
    Smooth broken power law.

    Parameters
    ----------
    x : array-like
        Input values. Must be positive.
    A : float
        Normalization. The function equals A at x = xb.
    xb : float
        Break location.
    alpha1 : float
        Power-law slope for x << xb.
    alpha2 : float
        Power-law slope for x >> xb.
    s : float
        Smoothness parameter. Larger s gives a sharper break.

    Returns
    -------
    y : array-like
        Smooth broken power-law values.

    Formula
    -------
    y = A * (x / xb)^(-alpha1) *
        [0.5 * (1 + (x / xb)^(1/s))]^{(alpha1 - alpha2) * s}
    """

    x = np.asarray(x, dtype=float)

    if np.any(x <= 0):
        raise ValueError("x must be positive.")
    if xb <= 0:
        raise ValueError("xb must be positive.")
    if s <= 0:
        raise ValueError("s must be positive.")

    xx = x / xb

    return A * xx**(-alpha1) * (0.5 * (1.0 + xx**(1.0 / s)))**((alpha1 - alpha2) * s)  # noqa: E501


def cl_turnover(ell):
    """
    CL fit to Alonso et al (2024), 2410.24134, Fig. 3
    """
    A = 0.94694
    ell_b = 73.952
    alpha1 = -0.1  # -0.85306
    alpha2 = 0.1  # 1.42731
    s = 0.59161

    return smooth_broken_power_law(ell, A=A, xb=ell_b,
                                   alpha1=alpha1, alpha2=alpha2, s=s)


def get_momentum_cl(lmax, out_dir, pl_index=2, std_offset=2, pl_index_v=3,
                    is_clustering=False, overwrite=False, pixwin=None):
    """
    Returns overdensity CL, velocity CL, and momentum CL.
    Ensures that overdensity has a standard deviation of 1/std_offset.
    Computes momentum CL using the moments method.
    """
    ls = np.arange(lmax+1)+0.01
    # clg = cl_turnover(ls)
    clg = 1/(10+ls)**pl_index  # Gaussian spectrum (unnormalized)
    norm = 4*np.pi/std_offset**2/np.sum((2*ls+1)*clg)
    # print('norm for overdensity cl', norm)
    # print("std (cl)", 1./std_offset)
    cl_od = norm*clg  # overdensity spectrum
    cl_od_pw = cl_od.copy()
    if pixwin is not None:
        cl_od_pw *= pixwin**2
    if is_clustering:
        return cl_od, cl_od, None, cl_od_pw, cl_od_pw
    cl_v = 1/(10+ls)**pl_index_v
    cl_th = cl_v + _evaluate_product_cl(lmax, cl_od, cl_v, out_dir,
                                        overwrite=overwrite)
    cl_th_pw = cl_v + _evaluate_product_cl(lmax, cl_od_pw, cl_v, out_dir,
                                           overwrite=overwrite, lab="pw")
    return cl_th, cl_od, cl_v, cl_th_pw, cl_od_pw


def get_bins_from_lmax_log(lmax, n=20, lmin=2):
    """
    Split the inclusive integer range [lmin, lmax] into log-scale integer bins.

    Returns
    -------
    bin_mins : np.ndarray
        Inclusive lower edge of each bin.
    bin_maxs : np.ndarray
        Inclusive upper edge of each bin.

    Guarantees
    ----------
    - Every integer from lmin to lmax is covered.
    - Bins are contiguous and non-overlapping.
    - No returned bin has bin_min == bin_max.
    - Therefore, if possible, each bin contains at least one integer.

    Notes
    -----
    - Supports lmin = 0 by using log(x + 1).
    - If the range has fewer than 2 integers, this guarantee is impossible.
    - The returned number of bins may be less than n.
    """
    lmax += 1  # To ensure that lmax is included.
    if n <= 0:
        raise ValueError("n must be positive")
    if lmin < 0:
        raise ValueError("lmin must be >= 0")
    if lmax < lmin:
        raise ValueError("lmax must be >= lmin")

    num_values = lmax - lmin + 1

    if num_values < 2:
        raise ValueError(
            "Cannot create bins with bin_min != bin_max when "
            "the range contains fewer than 2 integers"
        )

    # Need at least two integers per bin if bin_min != bin_max.
    n = min(n, num_values // 2)

    # Work in shifted coordinates so zero is allowed.
    lo = lmin + 1
    hi = lmax + 2  # exclusive upper edge, shifted by +1

    raw_edges = np.logspace(np.log10(lo), np.log10(hi), n + 1)

    # Convert back from shifted coordinates.
    edges = np.floor(raw_edges).astype(int) - 1

    # Enforce exact inclusive/exclusive endpoints.
    edges[0] = lmin
    edges[-1] = lmax + 1

    # Make edges strictly increasing.
    edges = np.maximum.accumulate(edges)

    # Enforce minimum bin width of 2 integers:
    # edge[i + 1] - edge[i] >= 2
    for i in range(1, len(edges)):
        min_allowed = edges[i - 1] + 2
        if edges[i] < min_allowed:
            edges[i] = min_allowed

    # If the forward pass pushed edges beyond the final endpoint,
    # repair from the right.
    edges[-1] = lmax + 1
    for i in range(len(edges) - 2, -1, -1):
        max_allowed = edges[i + 1] - 2
        if edges[i] > max_allowed:
            edges[i] = max_allowed

    # Re-enforce exact first endpoint.
    edges[0] = lmin

    bin_mins = edges[:-1]
    bin_maxs = edges[1:] - 1

    return nmt.NmtBin.from_edges(bin_mins, bin_maxs)


def _get_catalog_field(positions, alm, lmax, spin=0, mean=0., sigma=None,
                       seed=None, lmax_mask=None, fs=None, return_fs=False,
                       noise="gaussian", subtract_mean=True):
    """ Generates a NaMaster Catalog field from an alm,
    which we sample at the positions of the sources in cat.
    """
    if seed is not None:
        np.random.seed(seed)
    if alm is not None:
        fs = nmt.utils._alm2catalog_ducc0(alm, positions, spin=spin, lmax=lmax)
    elif fs is not None:
        pass
    else:
        raise ValueError("Must provide either alm or fs")
    if sigma is not None:
        if noise == "gaussian":
            fs += np.random.normal(np.full(fs.shape, mean), np.full(fs.shape, sigma))
        elif noise == "lognormal":
            # Taken from 
            s_sq = np.log(1.0 + sigma**2 / mean**2)
            m = np.log(mean) - 0.5 * s_sq
            ln_noise = np.random.lognormal(np.full(fs.shape, m),
                                           np.full(fs.shape, np.sqrt(s_sq)))
            fs += ln_noise
    # Noise mean subtraction
    if subtract_mean:
        fs -= np.mean(fs)
    if return_fs:
        return fs
    len = np.array(positions).shape[-1]
    f = nmt.NmtFieldCatalog(positions, np.ones(len), fs, lmax, spin=spin,
                            retain_catalog=True, lmax_mask=lmax_mask)
    return f


def _get_map_field(mask, map, alm, lmax, spin=0, lmax_mask=None):
    """ Generates a NaMaster field from a map or alm given a mask.
    """
    if map is None and alm is not None:
        pol = False if spin == 0 else True
        if pol and alm.shape[0] != 2:
            raise ValueError("Need QU alms as input.")
        nside = hp.npix2nside(len(mask))
        alms = np.array([0.*alm[0], alm[0], alm[1]])
        map = hp.alm2map(alms, nside=nside, pol=pol, lmax=lmax)
    elif map is None and alm is None:
        raise ValueError("map or alm must be given.")
    else:
        pol = False if spin == 0 else True
        if pol and map.shape[0] != 2:
            raise ValueError("Need QU maps as input.")
    if spin == 0:
        map = [map]

    return nmt.NmtField(mask, map, lmax=lmax, spin=spin, lmax_mask=lmax_mask)


def _get_momentum_field(positions, lmax, valm=None, mask=None,
                        positions_rand=None, spin=0, mean=0., sigma=0., seed=None,
                        lmax_mask=None, noise="gaussian", subtract_mean=True):
    """ Generates a NaMaster Catalog Momentum field from a catalog and a mask
    or a random catalog. If valm is not provided, this makes a Catalog
    Clustering field.
    """
    if mask is None and positions_rand is None:
        raise ValueError("Must provide mask or randoms.")
    if seed is not None:
        np.random.seed(seed)
    len = np.array(positions).shape[-1]
    weights = np.ones(len)
    weights_rand = None
    if positions_rand is not None:
        len_rand = np.array(positions_rand).shape[-1]
        weights_rand = np.ones(len_rand)
    if valm is None and sigma is None:
        f = nmt.NmtFieldCatalogClustering(
            positions, weights, positions_rand, weights_rand, lmax, mask=mask,
            retain_catalog=True, lmax_mask=lmax_mask)
    elif valm is not None:
        fs = nmt.utils._alm2catalog_ducc0(valm, positions,
                                          spin=spin, lmax=lmax)
        if sigma is not None:
            if noise == "gaussian":
                fs += np.random.normal(np.full(fs.shape, mean),
                                       np.full(fs.shape, sigma))
            elif noise == "lognormal":
                fs += np.random.lognormal(np.full(fs.shape, mean),
                                          np.full(fs.shape, sigma))
        # Noise mean subtraction
        if subtract_mean:
            fs -= np.mean(fs)

        f = nmt.NmtFieldCatalogMomentum(positions, weights, fs,
                                        positions_rand, weights_rand, lmax,
                                        mask=mask, spin=spin,
                                        retain_catalog=True,
                                        lmax_mask=lmax_mask)
    return f


def get_field(lmax, typ, cat=None, map=None, ran=None, msk=None, seed=None,
              lmax_mask=None, noise="gaussian", subtract_mean=True):
    """
    Parameters:
    lmax: int
        Maximum multipole used for power spectrum computations. 
    typ: str
        Field type. Can be "map", "cat", "num", "mom", to label map-based
        fields, tracer catalog fields, source clustering fields, or general
        momentum catalog fields (not source clustering), respectively.
    map: array
        Field mask, either in healpix or CAR. Needs corresponding randoms to
        be None. Assumed to have spin 0 if shape is [...] and spin 2
        if shape is [2, ...]. Ignored unless type is "map"
    cat: tuple of arrays (pos, flm, mean, sigma)
        Tuple containing arrays of source catalog positions, alms of the
        sampled field, and (optional) standard deviation of sampled field.
        flm are assumed to have spin 0 if ndim is 1 and spin 2
        if shape is [2, :]. Ignored if type is "map".
        flm is ignored if field type is "num". mean and sigma are ignored if
        None, or if type is "map" or "num".
    ran: array
        Array containing the random positions. Needs corresponding masks to be
        None. Ignored unless type is "num" or "mom".
    msk: array
        Field mask, either in healpix or CAR. Needs corresponding randoms to
        be None. Ignored if type is "cat".
    seed: int
        Random seed for sampling tracer-level noise. Ignored if None, and if
        type is "map" or "num.
    lmax_mask: int
        lmax considered for mode coupling matrix calculations
    noise: str
        Statistics assumed for uncorrelated noise at each source position.
        Accepted are "gaussiian", "lognormal". Defaults to "gaussian.
    subtract_mean: bool
        Whether to subtract the mean field value from all source values.
        Default: True
    Returns:
    fld: NmtField
        Output namaster field, of type NmtField, NmtFieldCatalog,
        NmtFieldCatalgClustering, NmtFieldCatalogMomentum if type is "map",
        "cat", "num", "mom", respectively.
    """
    if typ == "cat":
        if cat == "None":
            raise ValueError("Catalog must be provided.")
        pos, flm, mean, sigma = cat
        flm = np.array(flm)
        if flm.ndim == 1 or (flm.ndim == 2 and flm.shape[0] == 1):
            spin = 0
        elif flm.ndim == 2 and flm.shape[0] == 2:
            spin = 2
        else:
            raise ValueError("field alms have wrong shape.")
        return _get_catalog_field(pos, flm, lmax, spin=spin, mean=mean,
                                  sigma=sigma, seed=seed, lmax_mask=lmax_mask,
                                  noise=noise, subtract_mean=subtract_mean)
    elif typ == "map":
        map = np.array(map)
        if map is None:
            raise ValueError("Map must be provided.")
        if msk is None:
            raise ValueError("Mask must be provided.")
        if hasattr(map, "wcs"):
            if map.ndim == 2 or (map.ndim == 3 and map.shape[0] == 1):
                spin = 0
            elif map.ndim == 3 and map.shape[0] == 3:
                spin = 2
            else:
                raise ValueError("Map has wrong shape.")
        else:
            if map.ndim == 1 or (map.ndim == 2 and map.shape[0] == 1):
                spin = 0
            elif map.ndim == 2 and map.shape[0] == 2:
                spin = 2
            else:
                raise ValueError("Map has wrong shape.")
        return _get_map_field(msk, map, None, lmax, spin=spin,
                              lmax_mask=lmax_mask)
    elif typ in ["num", "mom"]:
        if cat is None:
            raise ValueError("Catalog must be provided.")
        if msk is None and ran is None:
            raise ValueError("Either mask or randoms must be provided")
        pos, flm, mean, sigma = cat
        pos_rand = None
        if ran is not None:
            msk = None
            pos_rand = ran
        if typ == "num":
            flm = None
            sigma = None
            mean = 0.
            spin = 0
        else:
            flm = np.array(flm)
            if flm.ndim == 1 or (flm.ndim == 2 and flm.shape[0] == 1):
                spin = 0
            elif flm.ndim == 2 and flm.shape[0] == 2:
                spin = 2
            else:
                raise ValueError("field alms have wrong shape.")
        return _get_momentum_field(pos, lmax, valm=flm, mask=msk,
                                   positions_rand=pos_rand, spin=spin,
                                   mean=mean, sigma=sigma, seed=seed,
                                   lmax_mask=lmax_mask,
                                   noise=noise, subtract_mean=subtract_mean)
    else:
        raise ValueError("Typ must be 'map', 'cat', 'mom', or 'num'")


def get_field_ids(types, spins, random_seeds):
    """
    Returns hash ids of 4 NmtFields given their field types, spins, and
    random seeds.
    """
    tsr_list = [(t, s, r) for t, s, r in zip(types, spins, random_seeds)]
    tsr_ids = [{v: k for k, v in enumerate(
        OrderedDict.fromkeys(tsr_list))}[tsr] for tsr in tsr_list]
    return tsr_ids


def get_mean_number_density_multipole(Nsrc, binary_mask=None):
    """
    Computes the mean particle number per square arcmin and the multipole
    corresponding to the mean innerparticle distance giben a binary mask.
    If binary_mask is None, the full sky is assumed. binary_mask must other-
    wise be a healpix map.
    """
    if binary_mask is None:
        numdens_isr = Nsrc / np.pi / 4.
    else:
        area_mask_sr = 4*np.pi*np.mean(binary_mask)
        numdens_isr = Nsrc / area_mask_sr
    numdens_iarcmin = numdens_isr * (np.pi/180./60.)**2
    ell_ipd = np.pi*np.sqrt(numdens_isr)

    return numdens_iarcmin, ell_ipd

import numpy as np
import healpy as hp
import pymaster as nmt
import os
from collections import OrderedDict


def get_catalog(npoints, m, seed):
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
    if npoints_get > 1e7:
        raise ValueError(f"Npints = {npoints_get}. "
                         "Input power spectrum is not steep enough.")
    phi = 2*np.pi*np.random.rand(npoints_get)
    th = np.arccos(-1+2*np.random.rand(npoints_get))
    ipix = hp.ang2pix(hp.npix2nside(npix), th, phi)
    mv = m[ipix]
    u = np.random.rand(npoints_get)
    keep = u <= mv
    th = th[keep]
    phi = phi[keep]
    return  np.array([th, phi], dtype=np.float64)


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


def gen_alms(cl, seed, spin=0):
    """
    """
    if cl is None:
        return None
    np.random.seed(seed)
    if spin == 0:
        return hp.synalm(cl)
    alm1 = hp.synalm(cl)
    np.random.seed(seed+12345)
    alm2 = hp.synalm(cl)
    return np.array([alm1, alm2])


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


def gen_cl_guess(cl0, spin1=0, spin2=0):
    """
    """
    if (spin1, spin2) == (0, 0):
        return np.array([cl0])
    elif spin1 == 0 or spin2 == 0:
        return np.array([cl0, cl0])
    else:
        return np.array([cl0, 0*cl0, 0*cl0, cl0])
    

def num_spin_comp(spin1, spin2):
    """
    Returns the number of spin components of a CL given spin pair.
    """
    if (spin1, spin2) == (0, 0):
        return 1
    elif spin1 == 0 or spin2 == 0:
        return 2
    return 4


def _evaluate_product_cl(lmax, cls_cc, cls_bb, plot_dir, overwrite=True):
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
    if os.path.isfile(f"{plot_dir}/theory_spectrum.npy") and not overwrite:
        return np.load(f"{plot_dir}/theory_spectrum.npy")

    from pyshtools.utils import Wigner3j  # noqa
    ells_w3j = np.arange(lmax+1)
    w3j = np.zeros_like(ells_w3j, dtype=float)
    big_w3j = np.zeros((lmax+1, lmax+1, lmax+1))
    for ell1 in ells_w3j[1:]:
        if ell1 % 100 == 0:
            print(" ", ell1)
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


def get_momentum_cl(lmax, out_dir, pl_index=2, std_offset=5, pl_index_v=3,
                    is_clustering=False, overwrite=False):
    """
    Returns overdensity CL, velocity CL, and momentum CL.
    Ensures that overdensity has a standard deviation of 1/std_offset.
    Computes momentum CL using the moments method.
    """
    if pl_index <= 1:
        raise ValueError("power law index for overdensity spetrum"
                         "must be strictly greater than 1.")
    ls = np.arange(lmax+1)
    clg = 1/(10+ls)**pl_index  # Gaussian spectrum (unnormalized)
    norm = 4*np.pi/std_offset**2/np.sum((2*ls+1)*clg)
    print('norm for overdensity cl', norm)
    print("std (cl)", 1./std_offset)
    cl_od = norm*clg  # overdensity spectrum
    if is_clustering:
        return cl_od, cl_od, None
    cl_v = 1/(10+ls)**pl_index_v
    cl_th = cl_v + _evaluate_product_cl(lmax, cl_od, cl_v, out_dir,
                                        overwrite=overwrite)
    return cl_th, cl_od, cl_v


def get_bins_from_lmax_log(lmax):
    """
    """
    ell_ini = np.array(
        [i for i in range(2, 10)] +
        [int(10**(n/5.)) for n in range(5, int(5*np.log10(lmax))+1)]
    )
    ell_end = np.array(
        [1+i for i in range(2, 10)] +
        [int(10**(n/5.)) for n in range(6, int(5*np.log10(lmax))+1)] + [lmax + 1]
    )
    return nmt.NmtBin.from_edges(ell_ini, ell_end)


def _get_catalog_field(positions, alm, lmax, spin=0):
    """ Generates a NaMaster Catalog field from an alm,
    which we sample at the positions of the sources in cat.
    """
    if alm is None:
        fs = None
    else:
        fs = nmt.utils._alm2catalog_ducc0(alm, positions,
                                          spin=spin, lmax=lmax)
    len = np.array(positions).shape[-1]
    f = nmt.NmtFieldCatalog(positions, np.ones(len),
                            fs, lmax, spin=spin, retain_catalog=True)
    return f


def _get_map_field(mask, map, alm, lmax, spin=0):
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

    return nmt.NmtField(mask, map, lmax=lmax, spin=spin)


def _get_momentum_field(positions, lmax, valm=None, mask=None,
                        positions_rand=None, spin=0):
    """ Generates a NaMaster Catalog Momentum field from a catalog and a mask
    or a random catalog. If valm is not provided, this makes a Catalog
    Clustering field.
    """
    if mask is None and positions_rand is None:
        raise ValueError("Must provide mask or randoms.")
    len = np.array(positions).shape[-1]
    weights = np.ones(len)
    weights_rand = None
    if positions_rand is not None:
        len_rand = np.array(positions_rand).shape[-1]
        weights_rand = np.ones(len_rand)
    if valm is None:
        f = nmt.NmtFieldCatalogClustering(
            positions, weights, positions_rand, weights_rand, lmax, mask=mask,
            retain_catalog=True)
    else:
        fs = nmt.utils._alm2catalog_ducc0(valm, positions,
                                          spin=spin, lmax=lmax)
        f = nmt.NmtFieldCatalogMomentum(positions, weights, fs,
                                        positions_rand, weights_rand, lmax,
                                        mask=mask, spin=spin,
                                        retain_catalog=True)
    return f


def get_field(lmax, typ, cat=None, map=None, ran=None, msk=None):
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
    cat: tuple of arrays (pos, flm)
        Tuple containing arrays of source catalog positions and alms of the
        sampled field. flm are assumed to have spin 0 if ndim is 1 and spin 2
        if shape is [2, :]. Ignored if type is "map".
        flm is ignored if field type is "num".
    ran: array
        Array containing the random positions. Needs corresponding masks to be
        None. Ignored unless type is "num" or "mom".
    msk: array
        Field mask, either in healpix or CAR. Needs corresponding randoms to
        be None. Ignored unless type is "num" or "mom".

    Returns:
    fld: NmtField
        Output namaster field, of type NmtField, NmtFieldCatalog,
        NmtFieldCatalgClustering, NmtFieldCatalogMomentum if type is "map",
        "cat", "num", "mom", respectively.
    """
    if typ == "cat":
        if cat == "None":
            raise ValueError("Catalog must be provided.")
        pos, flm = cat
        flm = np.array(flm)
        if flm.ndim == 1 or (flm.ndim == 2 and flm.shape[0] == 1):
            spin = 0
        elif flm.ndim == 2 and flm.shape[0] == 2:
            spin = 2
        else:
            raise ValueError("field alms have wrong shape.") 
        return _get_catalog_field(pos, flm, lmax, spin=spin)
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
        return _get_map_field(msk, map, None, lmax, spin=spin)
    elif typ in ["num", "mom"]:
        if cat is None:
            raise ValueError("Catalog must be provided.")
        if msk is None and ran is None:
            raise ValueError("Either mask or randoms must be provided")
        pos, flm = cat
        pos_rand = None
        if ran is not None:
            msk = None
            pos_rand = ran
        if typ == "num":
            flm = None
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
                                   positions_rand=pos_rand, spin=spin)
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
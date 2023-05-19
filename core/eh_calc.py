from math import pi
from core.eh_mics import iseven, split_str2numtext
import numpy as np
# TESTING jax library
# import jax.numpy as jnp
# import jax
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_enable_x64", True)
# Due to nature of vectorised integration and array size dependance of
# integration bounds, just-in-time compilaiton is difficult to implement.
# Without jit, jnp functions do not offer great improvements or can result even
# worse performance, when run on CPU or/and GPU.
# TODO - somehow modify functions to be able to use jit.
verbose = True


class Atom_orbitals:
    def __init__(self,
                 atom_index,
                 atom_type,
                 atom_coordinate,
                 orbital_configuration,
                 electron_configuration):
        """
        Parameters
        ----------
        atom_index : int
            Atom index in the molecule.
        atom_type : str
            Symbols of atoms (elements).
        atom_coordinate : ndarray
            Cartesian coordinates of an atom in a.u.
        orbital_configuration : dict, optional
            Orbital configurations of atoms. The default is orb_cfg.
        electron_configuration : dict, optional
            Electron configurations of atoms. Used for determening total count
            of electron of the system. The default is el_cfg.

        Returns
        -------
        None.

        """
        self.idx = atom_index
        self.a = atom_type
        self.crd = atom_coordinate
        self.orbital_cfg = orbital_configuration[atom_type]
        self.electron_cfg = electron_configuration[atom_type]
        self.orbitals = []
        self.ioniz_energ = []
        self.e_num = 0  # Total number of electrons.

    def alloc_orb(self, basis_set, ionization_energies_all):
        """
        Checks if provided system can be modelled. Creates tuple with wave
        function, parameters, atom coordinates and indexes. Also creates a
        tuple of ionization energies, which index corresponds to the atom of
        the same index.

        Parameters
        ----------
        basis_set : dict, optional
            Basis set for the atom. The default is bs.
        ionization_energies_all : dict, optional
            All ionization energies of atoms. The default is ie.

        Raises
        ------
        ValueError
            If atom is not supported.

        Returns
        -------
        None.

        """
        for o_layer in self.orbital_cfg:
            n_qn, l_qn, N = split_str2numtext(o_layer)  # qn - quantum number.
            if not iseven(N):  # Because of  EH, only full orbitals allowed.
                raise ValueError(
                    "Something went wrong with orbital configuration."
                    )
            if l_qn == "s":
                wf = (wf_s,)
            elif l_qn == "p":
                wf = (wf_pz, wf_py, wf_px)
            else:  # TODO - support for d-elements.
                raise ValueError(
                    "Higher orbitals are currently not supported."
                    )
            try:
                bs_param = basis_set[self.a][str(n_qn)+l_qn]
            except KeyError:  # TODO - support for all p elements.
                raise ValueError(
                    "Atom '"+self.a+"' is not supported."
                    )
            # Determening total count of electrons in atom.
            for electron_layer in self.electron_cfg:
                n2, l2, el_num = split_str2numtext(electron_layer)
                if n2 == n_qn and l2 == l_qn:
                    self.e_num += el_num
            for i in range(N//2):
                orbital = (wf[i], bs_param, self.crd, self.idx)
                self.orbitals.append(orbital)
                ie_val = ionization_energies_all[self.a][l_qn]
                self.ioniz_energ.append(ie_val)


def wf_s(x, y, z, c, r0):
    """
    S wavefunction, calculates amplitude.

    Parameters
    ----------
    x, y, z : 3d ndarray
        Coordinates of points.
    c : 2d ndarray
        Primitive gaussans parameters. First row for exponentials, the second
        for weighted sum.
    r : tuple
        Coordinates of an atom.

    Returns
    -------
    3d ndarray
        Amplitude at r0.

    """
    c = np.expand_dims(c, axis=(-1, -2, -3))
    x0, y0, z0 = r0
    r_sq = (x - x0)**2 + (y - y0)**2 + (z - z0)**2
    norm = (2 * c[0] / pi)**0.75
    return np.sum(c[1] * norm * np.exp(-c[0] * r_sq), axis=0)


def wf_px(x, y, z, c, r0):
    """
    Px wavefunction, calculates amplitude.

    Parameters
    ----------
    x, y, z : 3d ndarray
        Coordinates of points.
    c : 2d ndarray
        Primitive gaussans parameters. First row for exponentials, the second
        for weighted sum.
    r : tuple
        Coordinates of an atom.

    Returns
    -------
    3d ndarray
        Amplitude at r0.

    """
    c = np.expand_dims(c, axis=(-1, -2, -3))
    x0, y0, z0 = r0
    r_sq = (x - x0)**2 + (y - y0)**2 + (z - z0)**2
    norm = 2 * (2 / pi)**0.75 * c[0]**1.25
    return np.sum(c[1] * norm * np.exp(-c[0] * r_sq), axis=0) * (x - x0)


def wf_py(x, y, z, c, r0):
    """
    Py wavefunction, calculates amplitude.

    Parameters
    ----------
    x, y, z : 3d ndarray
        Coordinates of points.
    c : 2d ndarray
        Primitive gaussans parameters. First row for exponentials, the second
        for weighted sum.
    r : tuple
        Coordinates of an atom.

    Returns
    -------
    3d ndarray
        Amplitude at r0.

    """
    c = np.expand_dims(c, axis=(-1, -2, -3))
    x0, y0, z0 = r0
    r_sq = (x - x0)**2 + (y - y0)**2 + (z - z0)**2
    norm = 2 * (2 / pi)**0.75 * c[0]**1.25
    return np.sum(c[1] * norm * np.exp(-c[0] * r_sq), axis=0) * (y - y0)


def wf_pz(x, y, z, c, r0):
    """
    Pz wavefunction, calculates amplitude.

    Parameters
    ----------
    x, y, z : 3d ndarray
        Coordinates of points.
    c : 2d ndarray
        Primitive gaussans parameters. First row for exponentials, the second
        for weighted sum.
    r : tuple
        Coordinates of an atom.

    Returns
    -------
    3d ndarray
        Amplitude at r0.

    """
    c = np.expand_dims(c, axis=(-1, -2, -3))
    x0, y0, z0 = r0
    r_sq = (x - x0)**2 + (y - y0)**2 + (z - z0)**2
    norm = 2 * (2 / pi)**0.75 * c[0]**1.25
    return np.sum(c[1] * norm * np.exp(-c[0] * r_sq), axis=0) * (z - z0)


def f_multiply(z, y, x, orb1, orb2):
    """
    Multiplies provided function amplitudes.

    Parameters
    ----------
    x, y, z : 3d ndarray
        Coordinates of points.
    orb1 : tuple
        Contains function, ndarray of gaussain parameters and ndarray of atom
        coordinates.

    Returns
    -------
    3d ndarray
        Multiplied function amplitude at r0.

    """
    f1, c1, r1 = orb1
    f2, c2, r2 = orb2
    return f1(x, y, z, c1, r1) * f2(x, y, z, c2, r2)


def trap3d(x1, x2, y1, y2, z1, z2, h, orb1, orb2):
    """
    Integrates over 3D space with provided two functions.

    Parameters
    ----------
    x1, x2, y1, y2, z1, z2 : float
        Bounds of integration.
    h : float
        Integration step size, same for all dimensions.
    orb1, orb2 : tuple
        Contains function, ndarray of gaussain parameters and ndarray of atom
        coordinates.

    Returns
    -------
    integral3d : float
        Integration value.

    """
    nx = int((x2-x1)/h)
    ny = int((y2-y1)/h)
    nz = int((z2-z1)/h)
    x = np.linspace(x1, x2, nx)
    y = np.linspace(y1, y2, ny)
    z = np.linspace(z1, z2, nz)
    zz, yy, xx = np.meshgrid(z, y, x)
    val = f_multiply(zz, yy, xx, orb1, orb2)
    integral1d = np.trapz(val, xx)
    integral2d = np.trapz(integral1d, zz[:, :, 0])
    integral3d = np.trapz(integral2d, y)
    return integral3d

# jax.jit is not available, as mentioned in the beginning of this program.
# trap3d = jax.jit(trap3d, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8))


def overlap(orb1, orb2, cutoff, step_size):
    """
    Calculates overlap of two functions.

    Parameters
    ----------
    orb1, orb2 : tuple
        Contains function, ndarray of gaussain parameters, ndarray of atom
        coordinates and atom indexes.
    cutoff : float
        Value for adding/subtracting maximum/minimum value of two atom
        coordinates.
    step_size : float
        Integration step size, same for all dimensions.

    Returns
    -------
    TYPE
        Overlap value of provided two 3D functions.

    """
    if orb1[3] == orb2[3] and orb1[0] != orb2[0]:
        return 0  # Othogonal wave functions of the same atom.
    else:
        x1 = min((orb1[2][0], orb2[2][0])) - cutoff
        x2 = max((orb1[2][0], orb2[2][0])) + cutoff
        y1 = min((orb1[2][1], orb2[2][1])) - cutoff
        y2 = max((orb1[2][1], orb2[2][1])) + cutoff
        z1 = min((orb1[2][2], orb2[2][2])) - cutoff
        z2 = max((orb1[2][2], orb2[2][2])) + cutoff
        return trap3d(x1, x2, y1, y2, z1, z2, step_size, orb1[:3], orb2[:3])


def calc_H_S(orbitals, ioniz_energ, **kwargs):
    """
    Calculates Hamiltonian and overlap hermitian matrices.

    Parameters
    ----------
    orbitals : list
        Contains tuples of atoms, which can described with: Wave function,
        ndarray of gaussain parameters, ndarray of atom coordinates and atom
        indexes.
    ioniz_energ : list
        Contains ionization energies for each atom orbital.
    **kwargs : dict
        Can be cutoff, step_size, required for integration.

    Returns
    -------
    H_matrix : ndarray
        Hamiltonian matrix.
    S_matrix : ndarray
        Overlap matrix.

    """
    order = len(orbitals)
    S_matrix = np.ones((order, order), dtype=np.float32)
    H_matrix = np.ones((order, order), dtype=np.float32)
    for i in range(order):
        if verbose:
            percentage = 100 * i / (order - 1)
            print("Calulating overlap matrix %.1f" % percentage, r"%")
        for j in range(i, order):        
            if i == j:
                H_matrix[i][i] = -ioniz_energ[i]
            else:
                oi = orbitals[i]
                oj = orbitals[j]
                S_matrix[i][j] = overlap(oi, oj, **kwargs)
                Hi = ioniz_energ[i]
                Hj = ioniz_energ[j]
                H_matrix[i][j] = -1.75 * S_matrix[i][j] * (Hi + Hj) / 2
                H_matrix[j][i] = H_matrix[i][j]
                S_matrix[j][i] = S_matrix[i][j]
    return H_matrix, S_matrix


def solv_HC_SCe(H, S):
    """
    Solves HC=SCe matrix equation, where H, C, S are matrices and e is vector.

    Parameters
    ----------
    H : ndarray
        Hamiltonian matrix.
    S : ndarray
        Overlap matrix.

    Returns
    -------
    e : ndarray
        Energy vector of orbitals.
    C : ndarray
        Orbital coeficient matrix.

    """
    if verbose:
        print("Orthogonalizing overlap matrix...")
    D, U = np.linalg.eigh(S)
    D = np.sqrt(D)
    D = np.power(D, -1)
    D = np.diag(D)
    S = U.dot(D).dot(U.T)
    H = S.dot(H).dot(S)
    if verbose:
        print("Solving hamiltonian eigenvalue equation...")
    e, C = np.linalg.eigh(H)
    C = S.dot(C)
    return e, C


def total_energy(energy_vector, number_of_electrons):
    """
    Determines total energy of the system from orbital energies.

    Parameters
    ----------
    energy_vector : ndarray
        Energy vector of orbitals.
    number_of_electrons : int
        DESCRIPTION.

    Raises
    ------
    ValueError
        If provided electrons occupy unmodelled orbitals or electron count is
        negative.

    Returns
    -------
    E : float
        Electronic energy of the system.

    """
    E = 0
    if len(energy_vector) * 2 < number_of_electrons:
        raise ValueError("Charge is too negative. Not enough MOs.")
    elif number_of_electrons < 0:
        raise ValueError("Charge is too positive. Not enough MOs.")
    for i in range(number_of_electrons):
        E += energy_vector[i//2]
    return E

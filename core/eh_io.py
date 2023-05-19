import numpy as np
import pandas as pd
from core.eh_mics import eV2au, A2au, iseven
# pd.set_option("display.max_rows", None, "display.max_columns", None)
export_orbitals = True

def load_input_file(file_name):
    """
    Opens Gaussian 16 input file with extension .gjf. Extracts atom names, atom
    numbers and coordinates of the provided system. Coordinates must be in Å,
    in cartesian. Returned coordinates are in a.u.
    Programs returns a tuple, in which each element is represents an atom.

    Parameters
    ----------
    file_name : str
        String of the file name. Must have .gjf or no extension.

    Returns
    -------
    indexes : tuple
        A tuple of indexed atoms. Each atom is represented with unque index.
    names : tuple
        A tuple of all atom names in the file.
    coordinates : ndarray
        Array of all coordinates, each row is for each atom.

    """
    if not file_name.endswith('.gjf'):
        file_name += ".gjf"
    file = open(file_name)
    lines = []
    for line in file.readlines():
        lines.append(line.split())
    lines = list(filter(None, lines))
    lines.append([])  # Required for interating over coordinates.
    file.close()

    c = 0
    while c < len(lines):  # Searching for the beginning of coordinates.
        if len(lines[c]) > 3 and lines[c][0][0] != '#':
            crd_idx_start = c
            break
        c += 1
    while c < len(lines):  # Searching for the end of coordinates.
        c += 1
        if len(lines[c]) != len(lines[crd_idx_start]):
            crd_idx_end = c
            break

    lines = lines[crd_idx_start:crd_idx_end]

    indexes = (i for i in range(len(lines)))
    names = (a[0] for a in lines)
    coordinates = np.array(lines)[:, 1:].astype(np.float32) * A2au
    coordinates.flags["WRITEABLE"] = False
    return indexes, names, coordinates


def load_basis_set(file_name):
    """
    Opens Gaussian 16 basis set file with extension .gbs. Extracts atom names
    and parameters. Programs returns a dictionary, in which each parameter set
    is accessed with a key of atom name additional key of orbital type.

    Parameters
    ----------
    file_name : str
        String of the file name. Must have .gsb or no extension.

    Returns
    -------
    basis_set : dict
        A nested dictionary. Parameters are in ndarray format. First row is
        exponential parameters, and the second is weighted sum parameters.

    """
    if not file_name.endswith('.gbs'):
        file_name += ".gbs"
    bsdf = pd.read_csv(file_name, delim_whitespace=True, header=None)
    sto_n = int(bsdf.iloc[0, 1][4])  # STO-nG - number of primitive gaussians.
    bsdf = bsdf.iloc[1:, :3].reset_index(drop=True)
    # Finding location of parameters and atom names.
    s_loc = bsdf.index[bsdf[0] == 'S']
    sp_loc = bsdf.index[bsdf[0] == 'SP']
    atom_loc = s_loc - 1
    end_loc = bsdf.index[bsdf[0] == '****']
    atom_names = bsdf.iloc[atom_loc, 0]
    atom_names = atom_names.str.replace('-', '')

    basis_set = {}
    for i, j, e in zip(atom_loc, end_loc, atom_names):
        basis_set[e] = {}
        for s in s_loc:
            if s > i and s < j:
                bs_temp = bsdf.iloc[s + 1: s + 1 + sto_n, :2]
                bs_temp = bs_temp.replace('D', 'E', regex=True)
                basis_set[e]['1s'] = bs_temp.to_numpy(np.float32).T
                break
        for sp in sp_loc:
            if sp > i and sp < j:
                bs_temp = bsdf.iloc[sp + 1: sp + 1 + sto_n]
                bs_temp = bs_temp.replace('D', 'E', regex=True)
                bs_temp = bs_temp.to_numpy(np.float32).T
                # Creating view and not copying for memory efficiency.
                basis_set[e]['2s'] = bs_temp[:2]
                basis_set[e]['2p'] = bs_temp[::2]
                break
    return basis_set


def load_electron_config(file_name):
    """
    Opens user generated electron (or orbital) configuration file. File must
    have first column with element names, followed by all layers of electron
    (orbital) configuration separated with \t.

    Parameters
    ----------
    file_name : str
        String of the file name.

    Returns
    -------
    electron_config : dict
        A dictionary, in which keys are element names and values are lists of
        configuration layers.

    """
    file = open(file_name)
    electron_config = {}
    for line in file.read().splitlines():
        line = tuple(line.split('\t'))
        electron_config[line[0]] = line[1:]
    file.close()
    return electron_config


def load_ionisation_energies(file_name):
    """
    Opens YEAHMOP parameter file with extension .dat. Extracts atom names
    and ionization energies. Programs returns a dictionary, in which each
    parameter set is accessed with a key of atom name additional key of orbital
    type. Ionisation energies must be in eV and returned are in a.u.

    Parameters
    ----------
    file_name : str
        String of the file name. Must have .dat or no extension.

    Returns
    -------
    ionis_energ : dict
        A nested dictionary of ionization energies.

    """
    if not file_name.endswith('.dat'):
        file_name += ".dat"
    file = open(file_name)
    ionis_energ = {}
    for line in file.readlines():
        line = line.split()
        if line != []:
            if line[0][0] != ';':  # Filtering comments.
                e = line[0]
                if len(e) > 1:
                    e = e[0] + e[1:].lower()
                if e not in ionis_energ:
                    ionis_energ[e] = {}
                ionis_energ[e][line[5]] = abs(float(line[6])) * eV2au
    file.close()
    return ionis_energ


def save(file_name,
         input_file_name,
         charge,
         molecule_energy,
         orbitals_energy,
         electron_count):
    """
    Outputs file with results.

    Parameters
    ----------
    file_name : str
        Self explanatory.
    input_file_name : str
        Self explanatory.
    charge : int
        Charge of the molecule.
    molecule_energy : float
        Electronic molecule energy.
    orbitals_energy : ndarray
        Array of orbital energies.
    electron_count : int
        Self explanatory.

    Returns
    -------
    None.

    """
    file = open(file_name, 'a', encoding="utf-8")
    
    row1 = "Input file name: " + input_file_name
    
    row2 = "Charge: " + '{0:{1}}'.format(charge, '+' if charge else '')
    
    
    row3 = "Total electronic energy of the molecule: " + \
    "%.5f" % molecule_energy + " a.u."
    
    for r in (row1, row2, row3):
        file.write(r + "\n")
        
    if export_orbitals:
        file.write("Energy of orbitals:\n")
        e_count = len(orbitals_energy)
        column1 = ["\t" for i in range(e_count - electron_count // 2)] + \
        ["\tʌv" for i in range(electron_count // 2)]
        if not iseven(electron_count):
            column1[e_count - electron_count // 2 - 1] = "\tʌ "

        for i in range(e_count):
            row = str(e_count - i) + column1[i] + "\t"
            if orbitals_energy[e_count - i - 1] >= 0:
                row += ' '
            row += "%.5f" % orbitals_energy[e_count - i - 1]
            file.write(row + "\n")
    file.write("\n")
    file.close()

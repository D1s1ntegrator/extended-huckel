#!/usr/bin/env python3
import core.eh_io as eh_io
import core.eh_calc as eh_calc
from sys import argv
from time import time
# from eh_mics import au2eV

# Be loud and noisy if True.
eh_calc.verbose = True
eh_io.export_orbitals = True

# =============================================================================
#  Opening all input files and importing required data.
# =============================================================================

atoms = eh_io.load_input_file(argv[1])
bs = eh_io.load_basis_set(r'./parameters/sto3g.gbs')
orb_cfg = eh_io.load_electron_config(r'./parameters/orb_config_eh.dat')
el_cfg = eh_io.load_electron_config(r'./parameters/electron_config.dat')
ie = eh_io.load_ionisation_energies(r'./parameters/muller_parms.dat')
output_file_name = "output.log"

# =============================================================================
# Main part of the program
# =============================================================================

start = time()
if len(argv) < 3:
    argv.append(0)

total_electrons = -int(argv[2])
orbitals = []
ioniz_energ = []
if eh_calc.verbose:
    print("Initialization...")
for i, a, c in zip(*atoms):  # Weird behaviour, can't iterate "atoms" directly.
    ao = eh_calc.Atom_orbitals(i, a, c, orb_cfg, el_cfg)
    ao.alloc_orb(bs, ie)
    total_electrons += ao.e_num
    orbitals += ao.orbitals
    ioniz_energ += ao.ioniz_energ

H, S = eh_calc.calc_H_S(orbitals, ioniz_energ, cutoff=3, step_size=0.2)
e, C = eh_calc.solv_HC_SCe(H, S)

E = eh_calc.total_energy(e, total_electrons)

end = time()
time_diff = end - start
if eh_calc.verbose:
    print("Saving results to " + output_file_name)
print("Total runtime: %d min %.2f s" % (time_diff // 60, time_diff % 60))
eh_io.save(output_file_name, argv[1], int(argv[2]), E, e, total_electrons)

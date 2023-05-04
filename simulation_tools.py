from ase import Atoms
from abtem.structures import orthogonalize_cell
from ase.lattice.hexagonal import Graphene
from ase.visualize import view
from ase.parallel import paropen

from gpaw import GPAW, FermiDirac, PW, setup_paths
from gpaw.response.df import DielectricFunction
from gpaw.atom.generator import Generator
from gpaw.atom.configurations import parameters
from gpaw.xas import XAS

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import ndimage as ndi
import numpy as np
from math import sqrt, pi, sin, cos


def simulate_eels_ll(xtal, convergence_angle = 30, file_prefix = 'graphene'):

# Part 1: Ground state calculation

# Part 2: Find ground state density and diagonalize full hamiltonian
    calc = GPAW(mode=PW(500),
                kpts=(6, 6, 3),
                parallel={'domain': 1},
                # Use smaller Fermi-Dirac smearing to avoid intraband transitions:
                occupations=FermiDirac(0.05))

    xtal.calc = calc
    xtal.get_potential_energy()

    calc = calc.fixed_density(kpts = (20, 20, 7))

    # The result should also be converged with respect to bands:
    calc.diagonalize_full_hamiltonian(nbands=60)
    calc.write(str(file_prefix + '.gpw'), 'all')

    # Part 2: Spectra calculations
    f = paropen(str(file_prefix + '_q_list'), 'w')  # write q

    for i in range(1, 6):  # loop over different q: this is where the convergence angle will come in
        df = DielectricFunction(calc = str(file_prefix + '.gpw'),
                                frequencies={'type': 'nonlinear',
                                            'domega0': 0.01},
                                eta=0.2,  # Broadening parameter.
                                ecut=100,
                                # write different output for different q:
                                txt='out_df_%d.txt' % i)

        q_c = [i / 20.0, 0.0, 0.0]  # Gamma - M excitation

        df.get_eels_spectrum(q_c=q_c, filename = str(file_prefix + '_EELS_%d' % i))

        # Calculate cartesian momentum vector:
        cell_cv = xtal.get_cell()
        bcell_cv = 2 * np.pi * np.linalg.inv(cell_cv).T
        q_v = np.dot(q_c, bcell_cv)
        print(sqrt(np.inner(q_v, q_v)), file=f)

    f.close() 


def setup_core_states(xtal):
    setup_paths.insert(0, './core_states/')

    # Generate setups with 0.5, 1.0, 0.0 core holes in 1s
    elements = [sym for sym in np.unique(xtal.symbols)] #= ['O', 'C', 'N']
    coreholes = [0.5, 1.0, 0.0]
    names = ['hch1s', 'fch1s', 'xes1s']
    functionals = ['LDA', 'PBE']

    for el in elements:
        for name, ch in zip(names, coreholes):
            for funct in functionals:
                g = Generator(el, scalarrel=True, xcname=funct,
                            corehole=(1, 0, ch), nofiles=True)
                g.run(name=name, **parameters[el])


def simulate_eels_cl(): # xtal, file_prefix = 'graphene'
    setup_paths.insert(0, './core_states/') # /Users/austin/gpaw-setups-0.9.20000

    a = 12.0  # use a large cell
    d = 0.9575
    t = pi / 180 * 104.51
    atoms = Atoms('OH2',
                [(0, 0, 0),
                (d, 0, 0),
                (d * cos(t), d * sin(t), 0)],
                cell=(a, a, a))
    atoms.center()
    calc = GPAW(nbands=-30,
                h=0.2,
                txt='h2o_xas.txt',
                setups={'O': 'hch1s'})
    # the number of unoccupied stated will determine how
    # high you will get in energy

    atoms.calc = calc
    e = atoms.get_potential_energy()

    calc.write('h2o_xas.gpw')


    a = 12.0  # use a large cell

    d = 0.9575
    t = pi / 180 * 104.51
    atoms = Atoms('OH2',
                [(0, 0, 0),
                (d, 0, 0),
                (d * cos(t), d * sin(t), 0)],
                cell=(a, a, a))
    atoms.center()

    calc1 = GPAW(h=0.2,
                txt='h2o_gs.txt',
                xc='PBE')
    atoms.calc = calc1
    e1 = atoms.get_potential_energy() + calc1.get_reference_energy()

    calc2 = GPAW(h=0.2,
                txt='h2o_exc.txt',
                xc='PBE',
                charge=-1,
                spinpol=True,
                occupations=FermiDirac(0.0, fixmagmom=True),
                setups={0: 'fch1s'})
    atoms[0].magmom = 1
    atoms.calc = calc2
    e2 = atoms.get_potential_energy() + calc2.get_reference_energy()

    with paropen('dks.result', 'w') as fd:
        print('Energy difference:', e2 - e1, file=fd)


    dks_energy = 532.774  # from dks calcualtion

    calc = GPAW('h2o_xas.gpw')

    xas = XAS(calc, mode='xas')
    x, y = xas.get_spectra(fwhm=0.5, linbroad=[4.5, -1.0, 5.0])
    x_s, y_s = xas.get_spectra(stick=True)

    shift = dks_energy - x_s[0]  # shift the first transition

    y_tot = y[0] + y[1] + y[2]
    y_tot_s = y_s[0] + y_s[1] + y_s[2]

    plt.plot(x + shift, y_tot)
    plt.bar(x_s + shift, y_tot_s, width=0.05)
    plt.savefig('xas_h2o_spectrum.png')


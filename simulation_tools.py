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


def kpt_convergence(xtal, k_points = [2,4,6,8], cutoffs = [250, 300, 350]):
    # supply k_points and cutoff as arrays
    k_test_energy = []
    k_list, cutoff_list = [],[]

    for k in k_points:
        for c in cutoffs:
            kpts = (k,k,k)
            name = str('k_test' + str(k))

            calc = GPAW(mode=PW(c), kpts=kpts, txt=str(name) + '.txt') 
            xtal.calc = calc
            e = xtal.get_potential_energy()

            k_test_energy.append(e)
            k_list.append(k)
            cutoff_list.append(c)

    return k_test_energy, k_list, cutoff_list


def plot_psuedo_density(xtal, name, kpts = (4,4,4)):
    # Calculation
    calc = GPAW(mode=PW(300),       # cutoff
                kpts=kpts,     # k-points
                txt=name + '.txt')  # output file

    xtal.calc = calc
    energy = xtal.get_potential_energy()
    calc.write(name + '.gpw')
    density = calc.get_pseudo_density()

    # Plotting
    X, Y, Z = np.mgrid[0:density.shape[0], 0:density.shape[1], 0:density.shape[2]]
    theta = np.deg2rad(60)


    fig = go.Figure(data=go.Volume(
        x=X.flatten(), 
        y=Y.flatten(), 
        z=Z.flatten(),
        value=density.flatten(),
        #isomin=0.1,
        #isomax=0.8,
        opacity=0.1, # needs to be small to see through all surfaces
        surface_count=17, # needs to be a large number for good volume rendering
        ))

    x = xtal.get_scaled_positions()[:,0] * density.shape[0]
    y = xtal.get_scaled_positions()[:,1] * density.shape[1] 
    z = xtal.get_scaled_positions()[:,2] * density.shape[2]

    fig.add_scatter3d(x = x,y = y,z = z, mode = 'markers')

    fig.show()
    #fig.write_image(str("images/" + name + ".png"))


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
    setup_paths.insert(0, './')

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


def simulate_eels_cl(xtal, file_name, element):
    setup_paths.insert(0, './')

    calc = GPAW(nbands=-30,
                h=0.2,
                txt=str(file_name + '.txt'),
                setups={element: 'hch1s'})
    # the number of unoccupied stated will determine how high you will get in energy
    xtal.calc = calc
    e = xtal.get_potential_energy()
    calc.write(str(file_name +'_xas.gpw'))

    # ground state calc
    calc1 = GPAW(h=0.2,
                txt=str(file_name +'_gs.txt'),
                xc='PBE')
    xtal.calc = calc1
    e1 = xtal.get_potential_energy() + calc1.get_reference_energy()

    # excited state calc
    calc2 = GPAW(h=0.2,
                txt=str(file_name + '_exc.txt'),
                xc='PBE',
                charge=-1,
                spinpol=True,
                occupations=FermiDirac(0.0, fixmagmom=True),
                setups={0: 'fch1s'})
    xtal[0].magmom = 1
    xtal.calc = calc2
    e2 = xtal.get_potential_energy() + calc2.get_reference_energy()


    calc = GPAW(str(file_name + '_xas.gpw'))

    xas = XAS(calc, mode='xas')
    x, y = xas.get_spectra(fwhm=0.5, linbroad=[4.5, -1.0, 5.0])
    x_s, y_s = xas.get_spectra(stick=True)

    dks_energy = e2 - e1
    shift = dks_energy - x_s[0]  # shift the first transition

    y_tot = y[0] + y[1] + y[2]
    y_tot_s = y_s[0] + y_s[1] + y_s[2]

    plt.plot(x + shift, y_tot)
    #plt.bar(x_s + shift, y_tot_s, width=0.05)
    plt.savefig(str('xas_' + file_name + '_spectrum.png'))


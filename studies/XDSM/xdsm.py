from pyxdsm.XDSM import XDSM

opt = 'Optimization'
solver = 'MDA'
func = 'Function'

x = XDSM()

disciplines = [
    "Atmosphere",
    "Aerodynamics",
    "Stability",
    "Propulsion",
    "Power Systems",
    "Weights & Structures",
    "Dynamics"
]

# Disciplines
x.add_system('opt', opt, r'\text{IPOPT}')
x.add_system('atmo', func, r'\text{Atmosphere}')
x.add_system('aero', func, r'\text{Aerodynamics}')
x.add_system('stab', func, r'\text{Stability}')
x.add_system('prop', func, r'\text{Propulsion}')
x.add_system('power', func, r'\text{Power Systems}')
x.add_system('struct', func, r'\text{Structures \& Weights}')
x.add_system('dyn', func, r'\text{Dynamics}')

# Optimizer sends
x.connect('opt', 'atmo', r'\text{Traj.}')
x.connect('opt', 'aero', r'\text{A/C geom.}, \alpha, \delta')
x.connect('opt', 'stab', r'\text{A/C geom.}')
x.connect('opt', 'prop', r'\text{Prop. sizing}')
x.connect('opt', 'power', r'\text{Pow. sys. sizing}')
x.connect('opt', 'struct', r'\text{A/C geom.}')
x.connect('opt', 'dyn', r'\text{Traj.}')

# Internal sends
x.connect('atmo', 'aero', r'\rho, \nu')
x.connect('atmo', 'prop', r'\rho, \nu')
x.connect('atmo', 'dyn', r'\text{Winds}')
x.connect('aero', 'stab', 'M')
x.connect('aero', 'struct', 'L_{max}')
x.connect('aero', 'dyn', 'L, D')
x.connect('prop', 'power', 'P, P_{max}')
x.connect('prop', 'dyn', 'T')
x.connect('prop', 'struct', 'W_{prop}')
x.connect('power', 'struct', 'W_{psys}')
x.connect('power', 'dyn', 'P_{net}')
x.connect('struct', 'dyn', 'W_{total}')

# Optimizer returns
x.connect('aero', 'opt', r'\mathcal{R}(\Gamma)\text{ (for implicit solvers)}')
x.connect('stab', 'opt', r'\mathcal{R}(q, x_{sm})')
x.connect('power', 'opt', r'\mathcal{R}(\int P_{net} dt = E_{batt})')
x.connect('struct','opt', r'\mathcal{R}(W_{total} = \sum W_i)')
x.connect('dyn', 'opt', r'\mathcal{R}(F = ma)')

# IO
x.add_input('opt', r'\text{parameters}')
x.add_output('opt', r'x^* \text{ (A/C \& Mission)}')

x.write('out', build=False)

with open("out.tex", "w+") as f:
    f.write(
        r"""
                
        % XDSM diagram created with pyXDSM 2.1.1.
        \documentclass{article}
        \usepackage{geometry}
        \usepackage{amsfonts}
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{tikz}
        
        % Optional packages such as sfmath set through python interface
        \usepackage{sfmath}
        
        % Define the set of TikZ packages to be included in the architecture diagram document
        \usetikzlibrary{arrows,chains,positioning,scopes,shapes.geometric,shapes.misc,shadows}
        
        \geometry{legalpaper, landscape, margin=0.25in}
        
        % Set the border around all of the architecture diagrams to be tight to the diagrams themselves
        % (i.e. no longer need to tinker with page size parameters)
        %\usepackage[active,tightpage]{preview}
        %\PreviewEnvironment{tikzpicture}
        %\setlength{\PreviewBorder}{5pt}
        
        \begin{document}
        \input{"out.tikz"}
        \end{document}

        """
    )

import os

os.system("pdflatex out.tex")
os.remove("out.aux")
os.remove("out.log")
os.remove("out.tex")
os.remove("out.tikz")

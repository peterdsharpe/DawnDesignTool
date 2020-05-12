from pyxdsm.XDSM import XDSM

opt = 'Optimization'
solver = 'MDA'
func = 'Function'

x = XDSM()

def text(*strings):
    return [
        r'\text{%s}' % line
        for line in strings
    ]

# Disciplines
x.add_system('opt', opt, text("Optimizer:", "IPOPT"))
x.add_system('atmo', func, text("Atmosphere"))
x.add_system('aero', func, text("Aerodynamics"))
x.add_system('stab', func, text("Stability"))
x.add_system('prop', func, text("Propulsion"))
x.add_system('power', func, text("Power", "Systems"))
x.add_system('struct', func, text("Structures", "\& Weights"))
x.add_system('dyn', func, text("Dynamics"))

# Optimizer sends
x.connect('opt', 'atmo', text("Traj."))
x.connect('opt', 'aero', text("A/C geom.,",r"$\alpha, \delta$"))
x.connect('opt', 'stab', text("A/C geom."))
x.connect('opt', 'prop', text("Prop", "Sizing"))
x.connect('opt', 'power', text("Pow. Sys.", "Sizing"))
x.connect('opt', 'struct', text("A/C geom.,",r"$W_{total}$"))
x.connect('opt', 'dyn', text("Traj.,",r"$W_{total}$"))

# Internal sends
x.connect('atmo', 'aero', r'\rho, \nu')
x.connect('atmo', 'prop', r'\rho, \nu')
x.connect('atmo', 'dyn', text("Winds"))
x.connect('aero', 'stab', 'M')
x.connect('aero', 'struct', 'L_{max}')
x.connect('aero', 'dyn', 'L, D')
x.connect('prop', 'power', 'P, P_{max}')
x.connect('prop', 'dyn', 'T')
x.connect('prop', 'struct', 'W_{prop}')
x.connect('power', 'struct', 'W_{psys}')
x.connect('power', 'dyn', 'P_{net}')

# Optimizer returns
x.connect('aero', 'opt', text(r'$\mathcal{R}(\Gamma)$','(for implicit','solvers)'))
x.connect('stab', 'opt', r'\mathcal{R}(q, x_{sm})')
x.connect('power', 'opt', r'\mathcal{R}(\int P_{net} dt = E_{batt})')
x.connect('struct','opt', text(
    r'$\mathcal{R}(W_{total} = \sum W_i)$',
    r'$\mathcal{R}(\text{Beam Eqns.})$'
))
x.connect('dyn', 'opt', r'\mathcal{R}(F = ma)')

# IO
x.add_input('opt', text("Parameters"))
x.add_output('opt', text("Optimal","A/C \& Traj."))

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
# os.remove("out.tikz")

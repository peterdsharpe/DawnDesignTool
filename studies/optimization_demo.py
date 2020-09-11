import aerosandbox as asb

density = 1.225
velocity = 1
viscosity = 1.81e-5
CL = 0.4
pi = 3.14159

opti = asb.cas.Opti()

chord = opti.variable()
span = opti.variable()
opti.set_initial(chord, 1)
opti.set_initial(span, 1)

Re = density * velocity * chord / viscosity
CD_p = 1.328 * Re ** -0.5

AR = span / chord
CD_i = CL ** 2 / (pi * AR)

opti.subject_to(chord * span == 1)

opti.minimize(CD_p + CD_i)

opti.solver('ipopt')
sol = opti.solve()

print(sol.value(chord))
print(sol.value(span))
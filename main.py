from firedrake import UnitSquareMesh, VectorFunctionSpace, \
    FunctionSpace, TrialFunction, TestFunction, Expression, \
    DirichletBC, Function, Constant, inner, dx, grad, lhs, \
    rhs, div, assemble, File, solve

# Print log messages only from the root process in parallel
# parameters["std_out_all_processes"] = False;



# Set parameter values
T = 10.0
Re = 3000.0
Umax = 1.0
C=0.3

N=int((Re**(0.75)) * 2.0)

dt=C*(1.0/N)/Umax

mesh = UnitSquareMesh(N,N)

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)



# Define time-dependent pressure boundary condition
#p_in = Expression("sin(3.0*t)", t=0.0)

u_in = Expression(('1.0*(x[1]>0.5)+0.5*(x[1]<=0.5)', '0.0'))


# Define boundary conditions
noslip  = DirichletBC(V, (0, 0), [3, 4])
#inflow  = DirichletBC(Q, p_in, "x[0] < DOLFIN_EPS" )

inflow  = DirichletBC(V, u_in, 1)

outflow = DirichletBC(Q, 0, 2)

bcu = [noslip, inflow]
bcp = [outflow]

# Create functions
u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)

# Define coefficients
k = Constant(dt)
f = Constant((0, 0))

# Tentative velocity step
F1 = (1.0/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx + \
     (1.0/Re)*inner(grad(u), grad(v))*dx - inner(f, v)*dx

a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
a2 = inner(grad(p), grad(q))*dx
L2 = -(1/k)*div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx
L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Create files for storing solution
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")

# Time-stepping
t = dt

numSteps = int(T/dt)

for i in range(numSteps):

    # Update pressure boundary condition

    # Compute tentative velocity step
    # begin("Computing tentative velocity")
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, solver_parameters={"ksp_solver": "gmres", "pc_type": "jacobi"})
    # end()

    # Pressure correction
    # begin("Computing pressure correction")
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    solve(A2, p1.vector(), b2, solver_parameters={"ksp_solver": "cg", "pc_type": "hypre"})
    # end()

    # Velocity correction
    # begin("Computing velocity correction")
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, solver_parameters={"ksp_solver": "gmres", "pc_type": "jacobi"})
    # end()

    if i%(int(numSteps/1000)) == 0:
        # Save to file
        ufile.write(u1, time=t)
        pfile.write(p1, time=t)
        print("output!")

    # Move to next time step
    u0.assign(u1)
    t += dt
    print("t=%f" % t)


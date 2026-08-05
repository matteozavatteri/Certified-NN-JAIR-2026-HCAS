import gurobipy as gp
import conf

DELTA = 1e-3

# Property 1
def Pre1(m, inputs):
	pra, tau, rho, theta, psi = inputs
	m.addConstr(rho >= 55947.691)

def Post1(m, outputs):
	coc, wl, wr, sl, sr = outputs
	m.addConstr(coc + DELTA >= 1500)
	
# Property 2
def Pre2(m, inputs):
	pra, tau, rho, theta, psi = inputs
	m.addConstr(rho >= 55947.691)

def Post2(m, outputs):
	coc, wl, wr, sl, sr = outputs
	aMIN = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
	m.addConstr(aMIN == gp.min_([wl, wr, sl, sr]))
	m.addConstr(coc >= aMIN + DELTA)

# Property 3
def Pre3(m, inputs):
	pra, tau, rho, theta, psi = inputs
	m.addConstr(rho >= 1500)
	m.addConstr(rho <= 1800)
	m.addConstr(theta >= -0.06)
	m.addConstr(theta <= 0.06)
	m.addConstr(psi >= 3.10)

def Post3(m, outputs):
	coc, wl, wr, sl, sr = outputs
	aMAX = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
	m.addConstr(aMAX == gp.max_([wl, wr, sl, sr]))
	m.addConstr(coc + DELTA <= aMAX)

# Property 4
def Pre4(m, inputs):
	pra, tau, rho, theta, psi = inputs
	m.addConstr(rho >= 1500)
	m.addConstr(rho <= 1800)
	m.addConstr(theta >= -0.06)
	m.addConstr(theta <= 0.06)
	m.addConstr(psi == 0)

def Post4(m, outputs):
	coc, wl, wr, sl, sr = outputs
	aMAX = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
	m.addConstr(aMAX == gp.max_([wl, wr, sl, sr]))
	m.addConstr(coc + DELTA <= aMAX )

# Property 5
def Pre5(m, inputs):
	pra, tau, rho, theta, psi = inputs
	m.addConstr(rho >= 250)
	m.addConstr(rho <= 400)
	m.addConstr(theta >= 0.02)
	m.addConstr(theta <= 0.04)
	m.addConstr(psi >= -3.141592)
	m.addConstr(psi <= -3.141592+0.005)

def Post5(m, outputs):
	coc, wl, wr, sl, sr = outputs
	aMAX = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
	m.addConstr(aMAX == gp.max_([coc, wl, wr, sl]))
	m.addConstr(sr >= aMAX + DELTA)

# Property 6
def Pre6(m, inputs):
	pra, tau, rho, theta, psi = inputs
	m.addConstr(rho >= 12000)
	m.addConstr(rho <= 56000)
	a = m.addVar(vtype=gp.GRB.BINARY)
	b = m.addVar(vtype=gp.GRB.BINARY)
	m.addGenConstrIndicator(a, True, theta >= 0.07)
	m.addGenConstrIndicator(a, True, theta <= 3.141592)
	m.addGenConstrIndicator(b, True, theta >= -3.141592)
	m.addGenConstrIndicator(b, True, theta <= -0.07)
	m.addConstr(a+b >= 1)
	m.addConstr(psi >= -3.141592)
	m.addConstr(psi <= -3.141592+0.005)

def Post6(m, outputs):
	coc, wl, wr, sl, sr = outputs
	aMAX = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
	m.addConstr(aMAX == gp.max_([wl, wr, sl, sr]))
	m.addConstr(coc >= aMAX + DELTA)

# Property 7
def Pre7(m, inputs):
	pra, tau, rho, theta, psi = inputs
	m.addConstr(tau == 60)
	m.addConstr(rho >= 0)
	m.addConstr(rho <= 56000)
	m.addConstr(theta >= -3.141592)
	m.addConstr(theta <= 3.141592)
	m.addConstr(psi >= -3.141592)
	m.addConstr(psi <= 3.141592)

def Post7(m, outputs):
	coc, wl, wr, sl, sr = outputs
	aSL = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
	m.addConstr(aSL == gp.max_([coc, wl, wr, sr]))
	m.addConstr(sl + DELTA <= aSL)
	aSR = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
	m.addConstr(aSR == gp.max_([coc, wl, wr, sl]))
	m.addConstr(sr + DELTA <= aSR)

# Property 8
def Pre8(m, inputs):
	pra, tau, rho, theta, psi = inputs
	m.addConstr(pra == 1)
	m.addConstr(tau == 60)
	m.addConstr(rho >= 0)
	m.addConstr(rho <= 56000)
	m.addConstr(theta >= -3.141592)
	m.addConstr(theta <= -0.75*3.141592)
	m.addConstr(psi >= -0.1)
	m.addConstr(psi <= 0.1)

def Post8(m, outputs):
	coc, wl, wr, sl, sr = outputs

	aWL = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
	a = m.addVar(vtype=gp.GRB.BINARY)

	m.addConstr(aWL == gp.max_([coc, wr, sl, sr]))
	m.addGenConstrIndicator(a, True, wl >= aWL + DELTA)

	aCOC = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
	b = m.addVar(vtype=gp.GRB.BINARY)
	m.addConstr(aCOC == gp.max_([wl, wr, sl, sr]))
	m.addGenConstrIndicator(b, True, coc >= aCOC + DELTA )
	
	m.addConstr(a + b >= 1)

# Property 9
def Pre9(m, inputs):
	pra, tau, rho, theta, psi = inputs
	m.addConstr(pra == 2)
	m.addConstr(tau == 60)
	m.addConstr(rho >= 2000)
	m.addConstr(rho <= 7000)
	m.addConstr(theta >= -0.4)
	m.addConstr(theta <= -0.14)
	m.addConstr(psi >= -3.141592)
	m.addConstr(psi <= -3.141592+0.01)

def Post9(m, outputs):
	coc, wl, wr, sl, sr = outputs
	aMAX = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
	m.addConstr(aMAX == gp.max_([coc, wl, wr, sr]))
	m.addConstr(sl >= aMAX + DELTA)

# Property 10
def Pre10(m, inputs):
	pra, tau, rho, theta, psi = inputs
	m.addConstr(rho >= 36000)
	m.addConstr(rho <= 56000)
	m.addConstr(theta >= 0.7)
	m.addConstr(theta <= 3.141592)
	m.addConstr(psi >= -3.141592)
	m.addConstr(psi <= -3.141592+0.01)

def Post10(m, outputs):
	coc, wl, wr, sl, sr = outputs
	aMAX = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
	m.addConstr(aMAX == gp.max_([wl, wr, sl, sr]))
	m.addConstr(coc >= aMAX + DELTA)

def create_input_vars(m):
	pra   = m.addVar(lb=0, ub=4, vtype=gp.GRB.INTEGER)
	tau   = m.addVar(lb=0, ub=60, vtype=gp.GRB.INTEGER)
	rho   = m.addVar(lb=0, ub=56000, vtype=gp.GRB.CONTINUOUS)
	theta = m.addVar(lb=-3.141593, ub=3.141593, vtype=gp.GRB.CONTINUOUS)
	psi   = m.addVar(lb=-3.141593, ub=3.141593, vtype=gp.GRB.CONTINUOUS)
	return [pra, tau, rho, theta, psi]

def create_output_vars(m):
	O = [m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for i in range(conf.OutputSIZE)]
	return O


if __name__ == "__main__":

	PRE  = [Pre1, Pre2, Pre3, Pre4, Pre5, Pre6, Pre7, Pre8, Pre9, Pre10]
	POST = [Post1, Post2, Post3, Post4, Post5, Post6, Post7, Post8, Post9, Post10]

	env = gp.Env(empty=True)
	env.setParam('OutputFlag', 0)
	env.start()

	print("=============")
	print("2 properties")
	print("=============")
	for i in range(len(PRE)):
		for j in range(i+1,len(PRE)):
			m = gp.Model("model",env=env)
			inputs = create_input_vars(m)
			PRE[i](m, inputs)
			PRE[j](m, inputs)
			m.optimize()
			if m.Status != gp.GRB.OPTIMAL: # empty intersection of preconditions (realizable)
				print('(P{},P{}): trivially realizable'.format(i+1,j+1))
				m.dispose()
				continue

			outputs = create_output_vars(m)
			POST[i](m, outputs)
			POST[j](m, outputs)
			m.optimize()
			if m.Status == gp.GRB.INFEASIBLE: # non-empty intersection of preconditions but empty intersection of postconditions (irrealizable)
				print('(P{},P{}): irrealizable'.format(i+1,j+1))

			m.dispose()

	print("=============")
	print("3 properties")
	print("=============")
	for i in range(len(PRE)):
		for j in range(i+1,len(PRE)):
			for k in range(j+1,len(PRE)):
				m = gp.Model("model",env=env)
				inputs = create_input_vars(m)
				PRE[i](m, inputs)
				PRE[j](m, inputs)
				PRE[k](m, inputs)
				m.optimize()
				if m.Status == gp.GRB.INFEASIBLE:
					print('(P{},P{},P{}): pre-unsat'.format(i+1,j+1,k+1))
				elif m.Status == gp.GRB.OPTIMAL: # empty intersection of preconditions (realizable)
					outputs = create_output_vars(m)
					POST[i](m, outputs)
					POST[j](m, outputs)
					POST[k](m, outputs)
					m.optimize()
					if m.Status == gp.GRB.INFEASIBLE: # non-empty intersection of preconditions but empty intersection of postconditions (irrealizable)
						print('(P{},P{},P{}): irrealizable'.format(i+1,j+1,k+1))

			m.dispose()
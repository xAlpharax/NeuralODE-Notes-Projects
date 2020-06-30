##################################################################################################

#Code as from https://github.com/juliagusak/neural-ode-norm/tree/master/anode to get these solvers

##################################################################################################

import abc

class Time_Stepper(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, y0, Nt = 2 ):
        self.func = func
        self.Nt = Nt
        self.dt_next = []

    @abc.abstractmethod
    def step(self, func, t, dt, y):
        pass

    def integrate(self, y0):
        y1 = y0
        dt = 1. / float(self.Nt)
        for n in range(self.Nt):
            t0 = 0 + n * dt
            self.dt_next.append(dt)
            y1 = self.step(self.func, t0, dt, y1)

        return y1

#############################################################

class Euler(Time_Stepper):
    def step(self, func, t, dt, y):
        out = y + dt * func(t, y)
        return out

class RK2(Time_Stepper):
    def step(self, func, t, dt, y):
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k1)
        out = y + k2
        return out

class RK4(Time_Stepper):
    def step(self, func, t, dt, y):
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k1)
        k3 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k2)
        k4 = dt * func(t + dt, y + k3)
        out = y + 1.0 / 6.0 * k1 + 1.0 / 3.0 * k2 + 1.0 / 3.0 * k3 + 1.0 / 6.0 * k4
        return out

#############################################################

def odesolver(func, z0, options=None):
    if options is None:
        Nt = 2
    else:
        Nt = options['Nt']
    if options['method'] == 'Euler':
        solver = Euler(func, z0, Nt=Nt)
    elif options['method'] == 'RK2':
        solver = RK2(func, z0, Nt=Nt)
    elif options['method'] == 'RK4':
        solver = RK4(func, z0, Nt=Nt)
    else:
        print('error unsupported method passed')
        return

    z1 = solver.integrate(z0)

    if hasattr(func, 'base_func'):
        if hasattr(func.base_func, 'dt'):
            func.base_func.dt.append(solver.dt_next)
    elif hasattr(func, 'dt'):
        func.dt.append(solver.dt_next)

    return z1

#############################################################
from js import window, document, console
from pyodide.ffi import create_proxy

import numpy as np
from itertools import combinations
from scipy.spatial.distance import pdist

status = document.getElementById("status")

proxy = dict()
globals = dict()

globals['N'] = 3
globals['m'] = np.array([1, 2, 1])
globals['x'] = np.zeros((globals['N'], 2))
globals['p'] = np.zeros((globals['N'], 2))

M_pixel = np.zeros((2, 2))
globals['firstrun'] = True


def E():
    x = globals['x']
    p = globals['p']
    m = globals['m']

    m_combinations = list(combinations(range(3), 2))
    m_prod = np.array([m[i[0]] * m[i[1]] for i in m_combinations])
    E_kin = np.sum(np.sum(p**2, axis=1) / (2 * m))
    E_pot = np.sum(-m_prod / pdist(x))

    return E_kin + E_pot


def F(x):

    x0 = x[0, :]
    x1 = x[1, :]
    x2 = x[2, :]

    return np.array([
        -globals['m'][0] *
        (globals['m'][1] *
         (x0 - x1) / np.linalg.norm(x0 - x1)**3 + globals['m'][2] *
         (x0 - x2) / np.linalg.norm(x0 - x2)**3),
        -globals['m'][1] *
        (globals['m'][0] *
         (x1 - x0) / np.linalg.norm(x1 - x0)**3 + globals['m'][2] *
         (x1 - x2) / np.linalg.norm(x1 - x2)**3),
        -globals['m'][2] *
        (globals['m'][0] *
         (x2 - x0) / np.linalg.norm(x2 - x0)**3 + globals['m'][1] *
         (x2 - x1) / np.linalg.norm(x2 - x1)**3),
    ])


def verlet(F, x0, p0, m, dt):
    """
    Verlet integrator for one time step
    """
    x = x0
    p = p0

    p = p + 1 / 2 * F(x) * dt
    x = x + 1 / m[:, np.newaxis] * p * dt
    p = p + 1 / 2 * F(x) * dt

    return x, p


def draw_circle(ctx, x, y, r=5):

    pixelx = cwidth / 10 * (x + 5)
    pixely = cheight / 10 * (y + 5)

    ctx.beginPath()
    ctx.arc(int(pixelx), int(pixely), r, 0, 2 * np.pi)
    ctx.fillStyle = 'blue'
    ctx.fill()
    ctx.stroke()


def replot_canvas():

    dt = 1 / 25 / 10

    canvas = document.getElementById("canvas")
    ctx = canvas.getContext("2d")

    ctx.clearRect(0, 0, cwidth, cheight)

    x, p = verlet(F, globals['x'], globals['p'], globals['m'], dt)
    globals['x'] = x
    globals['p'] = p

    for i in range(globals['N']):
        draw_circle(ctx, x[i][0], x[i][1], 5 * globals['m'][i]**(1 / 3))

    L0 = float(np.cross(x[0, :], p[0, :]))
    L1 = float(np.cross(x[1, :], p[1, :]))
    L2 = float(np.cross(x[2, :], p[2, :]))
    L_tot = L0 + L1 + L2

    angular_mom = document.getElementById("angular_mom")
    s = f"L0 = {L0:.6f}Ẑ<br>L1 = {L1:.6f}Ẑ<br>L2 = {L2:.6f}Ẑ<br>L_tot = {L_tot:.6f}Ẑ<br><br>E = {E():.6f}"
    angular_mom.innerHTML = s


def push_queue(func, str):
    status.innerHTML = str
    proxy[func] = create_proxy(func)
    window.setTimeout(proxy[func], 100)


def set_main():

    console.log("threebody: entering set_main")

    global cwidth, cheight, pwidth

    canvas = document.getElementById("canvas")

    parentwidth = document.getElementById("parent").clientWidth
    cwidth = parentwidth
    cheight = int(cwidth * 0.75)

    pwidth = cwidth

    canvas.setAttribute("width", cwidth)
    canvas.setAttribute("height", cheight)

    console.log("threebody: exiting set_main")

    push_queue(init_ode, "&nbsp;")


def init_ode(foo=0):

    console.log("threebody: entering init_ode")

    if not globals['firstrun']:
        window.clearInterval(globals['timer'])

    exit_code = 0
    for i in range(3):
        error = document.getElementById("errorx" + str(i))
        try:
            globals['x'][i][0] = document.getElementById("x" + str(i)).value
            error.innerHTML = ""
        except ValueError:
            error.innerHTML = "Input error"
            status.innerHTML = ""
            exit_code = 1

    globals['p'][0][1] = 1
    globals['p'][1][1] = 0
    globals['p'][2][1] = -1

    if exit_code == 0:
        console.log("setting timer")
        proxy[replot_canvas] = create_proxy(replot_canvas)

        globals['timer'] = window.setInterval(proxy[replot_canvas], 40)

    if globals['firstrun']:
        for i in ["x0", "x1", "x2"]:
            ii = document.getElementById(i)
            ii.addEventListener("change", proxy[init_ode])

        globals['firstrun'] = 0


push_queue(set_main, "Setting up main window...")

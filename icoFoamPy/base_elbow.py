import numpy as np
import re


empty, fixedValue, noSlip, zeroGradient = 'empty', 'fixedValue', 'noSlip', 'zeroGradient'

def makeid(s):
    """使字符串符合标识符的要求，所有不合法的标识符符号会换成 "_" """
    res = re.sub("\W+", "_", s)
    if (res[0].isdigit()):
        return "_" + res
    return res

logdata = {}

def decode_hexfloat(v):
    """主要是为了转化十六进制浮点数"""
    if type(v) == list:
        return [decode_hexfloat(x) for x in v]
    elif type(v) == tuple:
        return [decode_hexfloat(x) for x in v]
    elif type(v) == str:
        if "0x" in v:
            return float.fromhex(v)
    return v

def handle(msg, k, v):
    """通过msg, 对k, v进行一些处理, 可以自行添加需要的msg"""
    k = makeid(k)
    v = decode_hexfloat(v)

    v_res = v
    def cat_lists(v):
        """拼接多个不同列表"""
        v_res = []
        for x in v:
            v_res += x
        return v_res
    if "boundary" in msg:
        v_res = cat_lists(v)
    if "Eqn" in msg:
        v_res[-1] = cat_lists(v_res[-1])
        v_res[-2] = cat_lists(v_res[-2])
        v_res = [np.array(x) for x in v_res]
        # v_res[-2] = v_res[-2][:,0] if v_res[-2].ndim == 2 else v_res[-2]
        if "laplacian" in msg:
            v_res[0] = v_res[2].copy() # laplacian lower = upper但是打印出来，总是显示没有lower
    else:
        v_res = np.array(v_res)
    
    return k, v_res



def step_by_step():
    """设置按一次Enter执行一步, 就是要有合适的输入才能从这个函数推出"""
    while True:
        # s = input("执行单步: [any], 执行剩下全部: all, 退出: exit, 暂停输入脚本: pause")
        s = input()
        if s == "all":
            global step_by_step
            step_by_step = lambda:None
            return
        elif s == "exit":
            exit()
        elif s == "pause":
            while True:
                s = input("输入脚本, 输入exit推出")
                if s == "exit":
                    break
                else:
                    exec(s)
        else:
            return
        

big_case = False
print_rtol_if_ok = False

def check(msg, var, foam_var, expected):
    """ 检查是否相等, 
    var: 这个python脚本里的变量名字符串
    foam_var: expected在icoFoam求解器里的名字字符串
    expected: 在icoFoam程序里实际得到的值
    """
    # if big_case and "tmp-var" in msg:
    #     return

    step_by_step()

    foam_var, expected = handle(msg, foam_var, expected)

    __log(foam_var, expected)

    var, var_name = eval(var), var

    # if "3->1" in msg:
    #     var = var[:, None]

    if allclose(var, expected):
        if not print_rtol_if_ok or "tmp-var" in msg:
            print(f"{var_name} vs {foam_var} ok")
        else:
            print(f"{var_name} vs {foam_var} ok:\nget {var_name}:\n{var}\nexpected {foam_var}:\n{expected}\nrtol between {var_name} and {foam_var}:")
            print(f"{diff(var,expected)}\n")
    else:
        print(f"update global {var_name} => {foam_var} rtol:")
        # if "boundaryField" not in var_name:
        #     print((var.shape, expected.shape))
        #     # print([*zip(var, expected, diff(var, expected))])
        print_diff(var, expected)
        eval(var_name)[:] = expected
    return var, expected

from collections.abc import Iterable

def print_diff(a, b, idx_base=()):
    """
    打印不同
    对于每个不同处，打印[索引idx, a[idx], b[idx], 相对误差]
    """
    try:
        if (isinstance(a, Iterable)):
            if type(a) in (list, tuple):
                assert(type(b) in (list, tuple) and len(a) == len(b))
            else:
                a, b = np.broadcast_arrays(a, b)
            if not isinstance(a[0], Iterable) or len(a[0]) == 3:
                for idx, is_close, xa, xb, rtol in zip(range(len(a)), np.isclose(a, b, atol=myatol, rtol=myrtol, equal_nan=True), a, b, diff(a, b)):
                    if not is_close.all():
                        print([*idx_base, idx], xa, xb, rtol)
            else:
                for i, x, y in zip(range(len(a)), a, b):
                    print_diff(x, y, tuple((*idx_base, i)))
        else:
            print("[:]", a, b, diff(a, b))
            
    except:
        print(type(a), type(b))
        raise


def __log(k, v):
    """不对k, v进行处理, 直接存储"""
    if big_case:
        return
    logdata.setdefault(k, [])
    logdata[k].append(v)


def log(msg, k, v):
    k, v = handle(msg, k, v)
    __log(k, v)


def set(msg, k, v):
    """设置全局变量, 添加名为k的全局变量, 其值为v"""
    k, v = handle(msg, k, v)
    __log(k, v)
    print("init global ", k)
    globals()[k] = v

logdata = {}



import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def diff(a, b):
    if type(a) == list or type(a) == tuple:
        return [diff(*item) for item in zip(a, b)]
    else:
        atol = np.abs(a-b)
        rtol = atol/(np.abs(b)+np.abs(b)+1e-100)*2
        return rtol
    
myrtol = 1e-100
myatol = 1e-10

def assert_allclose(x, y):
    if type(x) == list or type(x) == tuple:
        for item in zip(x, y):
            if not assert_allclose(*item):
                return False
        return True
    if not np.allclose(x, y, atol=myatol, rtol=myrtol, equal_nan=True):
        diff(x, y)
        raise Exception(f"diff between \n{x} \nand \n{y} \nis \n{diff(x,y)}")
    return True

def allclose(x, y):
    if type(x) == list or type(x) == tuple:
        for item in zip(x, y):
            if not allclose(*item):
                return False
        return True
    return np.allclose(x, y, atol=myatol, rtol=myrtol, equal_nan=True)


def init_variable():
    if 'face_owner' in globals():
        return
    global face_owner, face_neighbour, face_area, face_n, detla, detla_norm, Volumes, count_cell, count_internal_face, count_boundary_face, nc, ni, nb, face_area_norm, face_area_internal, face_area_boundary, dt, nu, d_cell, detla_boundary, d_cell_bounday, boundaryField, U_boundary, surfaceInterpolation_weights
    face_owner = mesh_faceOwner_
    face_neighbour = mesh_faceNeighbour_
    face_area = np.array(mesh_faceAreas_)
    detla = np.array(mesh_delta_ref_)
    detla
    detla_norm = np.linalg.norm(detla, axis=1)
    Volumes = np.array(mesh_cellVolumes_)
    Volumes
    count_cell = len(Volumes)
    count_internal_face =  len(face_neighbour)
    count_boundary_face = len(U_boundaryField_)
    nc, ni, nb = count_cell, count_internal_face, count_boundary_face
    face_area_norm = np.linalg.norm(face_area, axis=1)
    face_area_internal = face_area[:count_internal_face]
    face_area_boundary = face_area[ni:ni+nb]
    face_n = face_area / face_area_norm[:,None]
    count_boundary_face
    dt = runTime_deltaT_value_
    nu = nu_value_
    d_cell = detla_norm
    d_cell
    detla_boundary = np.array(mesh_delta_ref_boundaryField_)
    d_cell_bounday = np.linalg.norm(detla_boundary, axis=1)
    boundaryField = face_owner[count_internal_face: count_internal_face + count_boundary_face]
    U_boundary = U_boundaryField_
    if "phi_mesh_surfaceInterpolation_weights_" not in globals():
        surfaceInterpolation_weights = np.empty(ni)
        surfaceInterpolation_weights.fill(0.5)
    else:
        surfaceInterpolation_weights = phi_mesh_surfaceInterpolation_weights_


########################################
# fun
########################################




def sum_lower_upper(UEqn):
    upper, lower = UEqn[UPPER], UEqn[LOWER]
    res = np.zeros(nc)
    for face in range(len(face_neighbour)):
        p, n = face_owner[face], face_neighbour[face]
        res[n] += upper[face]
        res[p] += lower[face]
    return res

# UEqn
def makeMtx(UEqn):
    lower, diag, upper, source, vic, vbc = UEqn
    Mtx = np.zeros((nc,nc))
    source = source.copy()
    Mtx[[*range(nc)],[*range(nc)]]=diag
    for index in range(ni):
        i, j = face_owner[index], face_neighbour[index]
        Mtx[i][j] = upper[index]
        Mtx[j][i] = lower[index]
    for i in range(nb):
        cell_owner = boundaryField[i]
        source[cell_owner] = source[cell_owner] + vbc[i]
        Mtx[cell_owner][cell_owner] += vic[i]
    return Mtx, source

def getH(div_phi_U):
    tmp, source = (makeMtx(div_phi_U))
    tmp[[*range(nc)],[*range(nc)]]=0
    tmp = -np.matmul(tmp, U)
    tmp = tmp
#     print(source_bound)
    tmp += source
    tmp /= Volumes[:,None]
    return tmp

def getA(laplacian_nu_U):
    tmp = laplacian_nu_U[1].copy()
#     print(tmp/V)
#     print(laplacian_nu_U[4])
#     print(boundaryField)
    for i in range(nb):
#         print(boundaryField[i], laplacian_nu_U[4][i])
        if type(laplacian_nu_U[4]) == float or type(laplacian_nu_U[4]) == int:
            tmp[boundaryField[i]] += laplacian_nu_U[INTERNALCOEFFS]
        else:
            tmp[boundaryField[i]] += laplacian_nu_U[INTERNALCOEFFS][i]
    return tmp/Volumes
# print(*tmp,sep="\n")
# diff(tmp, str2arr("((-0.997595 1.31939 0) (-0.980441 -1.0758 0) (-26.5416 -6.17941 0) (-22.8266 3.59791 0) (180.021 -13.4631 0) (278.541 59.5667 0));"))
# getA(laplacian_nu_U) # 预期(-3.9 -3.9 -3 -3 -3.9 -3.9)


LOWER, DIAG, UPPER, SOURCE, INTERNALCOEFFS, BOUNDARYCOEFFS = range(6)

def make_UEqn():
    init_variable()
    global ddt_U, div_phi_U, laplacian_nu_U, UEqn
    ddt_U = [0, Volumes / dt, 0, Volumes[:,None] / dt * U, 0, 0]


    div_phi_U = [0]*6
    div_phi_U[UPPER] = phi * (1-surfaceInterpolation_weights)
    div_phi_U[LOWER] = -phi * (surfaceInterpolation_weights)
    div_phi_U[DIAG] = -sum_lower_upper(div_phi_U) 
    div_phi_U[BOUNDARYCOEFFS] = -np.sum(U_boundary * face_area_boundary, axis=1)[:,None]*U_boundary
    div_phi_U[BOUNDARYCOEFFS][boundary_out_begin:boundary_out_end] = 0
    div_phi_U[INTERNALCOEFFS] = np.zeros((nb, 3))
    for face in range(boundary_out_begin, boundary_out_end):
        div_phi_U[INTERNALCOEFFS][face] = phi_boundaryField_[face]
        # print(U[face_owner[face+ni]])
    # print(f"{logdata['U_boundaryField_'][-1][boundary_out_begin:boundary_out_end]=}")
    # if (len(logdata['U_boundaryField_'])) >= 2:
        # print(f"{logdata['U_boundaryField_'][-2][boundary_out_begin:boundary_out_end]=}")
    laplacian_nu_U = [0]*6

    laplacian_nu_U[UPPER] = nu * np.linalg.norm(face_area[:ni], axis=1)/d_cell
    laplacian_nu_U[LOWER] = laplacian_nu_U[UPPER].copy()
    laplacian_nu_U[DIAG] = -sum_lower_upper(laplacian_nu_U)

    
    laplacian_nu_U[INTERNALCOEFFS] = -(nu * np.linalg.norm(face_area[ni:ni+nb], axis=1) / d_cell_bounday)[:,None]
    laplacian_nu_U[INTERNALCOEFFS][boundary_out_begin:boundary_out_end] = 0
    laplacian_nu_U[BOUNDARYCOEFFS] = -(nu * np.linalg.norm(face_area[ni:ni+nb], axis=1) / d_cell_bounday)[:,None]*U_boundary
    laplacian_nu_U[BOUNDARYCOEFFS][boundary_out_begin:boundary_out_end] = 0
    # print(laplacian_nu_U[INTERNALCOEFFS])

    for Eqn in (ddt_U, div_phi_U, laplacian_nu_U):
        for x, y in zip(range(nc), (ni, nc, ni, (nc,3), (nb,3), (nb,3))):
            Eqn[x] = Eqn[x] + np.zeros(y)

    _UEqn = lambda: [x[0]+x[1]-x[2] for x in zip(ddt_U, div_phi_U, laplacian_nu_U)]
    
    UEqn = _UEqn()
    


def make_U_momentumPredictor():
    
    global grad_p, UEqn_grad, U_momentumPredictor, U, U_old, phi_old
    grad_p = fvc_grad(p)
    # print(UEqn[SOURCE])
    UEqn_grad = UEqn.copy()
    UEqn_grad[SOURCE] = UEqn_grad[SOURCE] - grad_p * Volumes[:,None] # 不使用+=，防止对UEqn进行修改
    UEqn_grad[INTERNALCOEFFS] = UEqn_grad[INTERNALCOEFFS][:,0] # 同上
    U_momentumPredictor = np.linalg.solve(*makeMtx(UEqn_grad))
    U, U_old = U_momentumPredictor, U
    phi_old = phi # phi_old似乎是始终和phi相同
#     print(f"{U=},{U_old=}")
#     print(U)

    
def flux(vol_f):
    res = np.zeros(ni)
    
    for face in range(ni):
        p, n = face_owner[face], face_neighbour[face]
        w = surfaceInterpolation_weights[face]
        face_f = vol_f[n]*(1-w)+vol_f[p]*(w)
        if face_f.size == 1:
            res[face] = face_area_norm[face] * face_f
        else:
            res[face] = np.matmul(face_area[face], face_f)
    return res

def interpolate(U):
    res = np.zeros((ni, *U.shape[1:]))
    for index in range(ni):
        i, j = face_owner[index], face_neighbour[index]
        w = surfaceInterpolation_weights[index]
        # res[index] = (U[i]+U[j])/2
        res[index] = U[i]*w+(1-w)*U[j]
    return res

def laplacian_base(k):
    res = [np.zeros(s) for s in (ni, nc, ni)]
    res += [0,0,0]
    k_arr = np.zeros(nc)
    k_arr[:] = k
    k_surface = interpolate(k_arr)
#     print(k)
#     print(k_surface)
    res[LOWER] = face_area_norm[:ni] / detla_norm * k_surface
    res[UPPER] = face_area_norm[:ni] / detla_norm * k_surface
    for idx in range(len(face_neighbour)):
        p, n = face_owner[idx], face_neighbour[idx] 
        res[DIAG][n] -= res[UPPER][idx]
        res[DIAG][p] -= res[LOWER][idx]
    return res

def div_phi(phi, phi_bound):
    res = np.zeros((nc, *phi.shape[1:]))
    for idx in range(len(face_neighbour)):
        p, n = face_owner[idx], face_neighbour[idx]
        res[p] += phi[idx]
        res[n] -= phi[idx]
    for bound in range(nb):
        # print(ni+bound, np.matmul(U_boundary[bound], face_area_boundary[bound]))
        res[face_owner[ni+bound]] += phi_bound[bound]
    # print(f"{U_boundary}")
    # print(f"{face_area_boundary}")

#     print(res)
    # res /= Volumes # 不知道为什么
    return res

def faceH(Eqn):
    res = np.zeros(ni)
    for face in range(ni):
        res[face] = Eqn[UPPER][face] * p[face_neighbour[face]] - Eqn[LOWER][face] * p[face_owner[face]]
    return res

def fvc_grad(p):
    res = np.zeros((nc, 3))
    p_phi_arr = interpolate(p)
    for face in range(ni):
        w = surfaceInterpolation_weights[face]
        p_phi = p_phi_arr[face] * face_area[face]
        res[face_owner[face]] += p_phi
        res[face_neighbour[face]] -= p_phi
    for face in range(ni, ni+nb):
        if boundary_out_begin <= face-ni < boundary_out_end:
            pass
        else:
            res[face_owner[face]] += p[face_owner[face]] * face_area[face] # 边界面的压力插强为从属cell的压强
    return res / Volumes[:, None]


def setReference(celli=0, value=0.0):
    pEqn[SOURCE][celli] += pEqn[DIAG][celli]*value;
    pEqn[DIAG][celli] += pEqn[DIAG][celli];

    
def make_pUqn():
    global rAU, HbyA, U_old, U, phiCorr, gama, ddtCorr, interpolate_rAU, phiHbyA, pEqn, div_phiHbyA, pEqn_flux, flux_HbyA, phiHbyA_boundaryField_
    UEqn_local = UEqn.copy()
    UEqn_local[INTERNALCOEFFS] = UEqn_local[INTERNALCOEFFS][:,0] 
    rAU = 1/getA(UEqn_local)
    HbyA = rAU[:, None] * getH(UEqn_local)
    global HbyA_boundary
    HbyA_boundary = np.zeros((nb,3))
    HbyA_boundary[:] = U_boundary
    for face_b in range(boundary_out_begin, boundary_out_end):
        HbyA_boundary[face_b] = HbyA[face_owner[face_b+ni]]

    phiCorr = phi_old - flux(U_old) # phi_old 和phi相同
    gama = 1 - np.minimum(np.abs(phiCorr/(phi_old+1e-100)), 1)
    ddtCorr = gama * phiCorr / dt
    interpolate_rAU = interpolate(rAU)
    flux_HbyA = flux(HbyA)
    phiHbyA = flux_HbyA + interpolate_rAU * ddtCorr
#     phiHbyA = str2arr("(8.3513e-06 0.000120961 -3.6998e-05 -4.27049e-05 -0.00036813 0.000166063 -0.000450094)")

    pEqn = laplacian_base(rAU)
    
    phiHbyA_boundaryField_ = (HbyA_boundary * face_area[ni: ni+nb])[:,1]
    phiHbyA_boundaryField_ = phi_boundaryField_
    phi_boundaryField_[boundary_out_begin:boundary_out_end] = ((HbyA_boundary * face_area[ni: ni+nb])[:,1])[boundary_out_begin:boundary_out_end]
    div_phiHbyA = div_phi(phiHbyA, phiHbyA_boundaryField_)
    pEqn[SOURCE] = div_phiHbyA

    # assert_allclose(pEqn[SOURCE], str2arr("(0.000129312 -4.53493e-05 -0.000531795 0.000245766 0.000418036 -0.000215969)"))
    pEqn[INTERNALCOEFFS] = np.zeros(nb)
    pEqn_VALUEINTERNALCOEFFS = -(np.linalg.norm(face_area[ni:ni+nb], axis=1) / d_cell_bounday)
    for face in range(boundary_out_begin, boundary_out_end):
        pEqn_VALUEINTERNALCOEFFS[face] *= rAU[face_owner[ni+face]]
    pEqn[INTERNALCOEFFS][boundary_out_begin:boundary_out_end] = pEqn_VALUEINTERNALCOEFFS[boundary_out_begin:boundary_out_end]
    
    pEqn[BOUNDARYCOEFFS] = np.zeros(nb)

#         phiHbyA_expected = str2arr("(8.3513e-06 0.000120961 -3.6998e-05 -4.27049e-05 -0.00036813 0.000166063 -0.000450094)")
#         assert_allclose(div_phi(phiHbyA_expected)/Volumes, str2arr("(7.75872 -2.72096 -31.9077 14.7459 25.0821 -12.9581)"))

#         assert_allclose(np.linalg.solve(*makeMtx(pEqn))+1.20714e-03-19.17651827, str2arr("(0.00120714 -0.100926 1.80044 -0.761193 -2.30927 0.824968)"))

   

def slove_p():
    global p_star, pEqn_flux, U_2, phi_old, phi, U_old, U, p, p_old 
    p_star = np.linalg.solve(*makeMtx(pEqn))
#     p_star += [0.0198326,  7.89335e-07][piso_loop]
#         print("p_star", p_star)
    # assert_allclose(p_star, str2arr("(0.000469259 -0.10172 1.80042 -0.761427 -2.30892 0.824955)"))
    p, p_old = p_star, p

def update_pu():
    global p_star, pEqn_flux, U_2, phi_old, phi, U_old, U, p, p_old, phi_new, fvc_grad_p, U_boundary

    pEqn_flux = faceH(pEqn)

    phi_new = phiHbyA - pEqn_flux
    fvc_grad_p = fvc_grad(p)
    U_2 = HbyA - rAU.reshape(nc,1)*fvc_grad_p
    phi = phi_new
#     U_old, U = U, U_2
    U = U_2 # 这里不更新U_old
    for face in range(boundary_out_begin, boundary_out_end):
        U_boundary[face] = U[face_owner[ni+face]]

    update_phi_boundary(phi, phi_boundaryField_)

def update_phi_boundary(phi, phi_boundaryField_):
    phi_boundaryField_[boundary_out_begin: boundary_out_end] = 0
    for face_b in range(boundary_out_begin, boundary_out_end):
        cell = face_owner[ni+face_b]
        for face_i in range(ni):
            if face_owner[face_i] == cell:
                phi_boundaryField_[face_b] += -phi[face_i]
                # print(face_b, -phi[face_i])
            elif face_neighbour[face_i] == cell:
                phi_boundaryField_[face_b] += phi[face_i]

#     print(f"{U=},{U_old=}")
    
################################################################
################################################################


big_case = True
print_rtol_if_ok = False
# myrtol = 1e-4

auto = False

if not auto:
    while True:
        s = input("手动控制执行?[[y]/n]\n")
        if s == "y" or s == "":
            print("Enter执行下一步, exit退出\n")
            break
        elif s =="n":
            step_by_step = lambda:None
            break
else:
    step_by_step = lambda:None

def run_times_control(times, fun):
    count = 0
    def decorated_fun(*args, **kargs):
        count += 1
        if count > times:
            print(f"run this function for {times} already. exit!")
            exit()
        return fun(*args, **kargs)
    
    return fun
make_pUqn = run_times_control(3, make_pUqn)

# 出口边界，因为只有出口边界是特殊的，所以暂时这样处理，后续实现通用的方案
boundary_out_begin, boundary_out_end = 112, 120 # 出口边界的范围额
# boundary_out_begin, boundary_out_end = 0, 0 # 使用cavity


myrtol = 1e-6
myatol = 1e-8

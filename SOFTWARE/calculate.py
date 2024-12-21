import numpy as np
from math import *
import matplotlib.pyplot as plt
import cmath

R = 8.314
def ALPHA(omega,T,T_c):
	"""
	alpha(T)函数，传入偏心因子，对比温度
	"""
	T_r = T/T_c
	return (1+(0.48+1.574*omega-0.176*omega*omega)*(1-sqrt(T_r)))**2
def A_srk(T,T_c,p_c,omega):
	return 0.42748*ALPHA(omega,T,T_c)*(R**2*T_c**2)/(p_c)
def B_srk(T_c,p_c):
	return 0.08664*R*T_c/p_c
def SOLVE_THREE(P,Q)->list:
	OMEGA = (-1+(sqrt(3))*1j)/(2)
	l1,l2 = -Q/2+cmath.sqrt((Q/2)**2+(P/3)**3),-Q/2-cmath.sqrt((Q/2)**2+(P/3)**3)
	X1 = l1**(1/3)+l2**(1/3)
	X2 = OMEGA*l1**(1/3)+OMEGA*OMEGA*l2**(1/3)
	X3 = OMEGA*OMEGA*l1**(1/3)+OMEGA*l2**(1/3)
	X = [X1,X2,X3]
	solve_list = []
	for num in X:
		if num.imag == 0: # 实数
			solve_list.append(num.real)
	return solve_list
def round_ri(xo, n=4):
	xr, xi = round(xo.real, n), round(xo.imag, n)
	if xi == 0:
		return xr
	else:
		return complex(xr, xi)
def cardano_solution(a, b, c, d):
	#u = round((9*a*b*c-27*(a**2)*d-2*(b**3)) / (54*(a**3)), 4)
	#v = round(3*(4*a*c**3 - b**2*c**2-18*a*b*c*d+27*a**2*d**2+4*b**3*d) / (18**2*a**4), 4) ** (1.0/2)
	u = (9*a*b*c-27*(a**2)*d-2*(b**3)) / (54*(a**3))
	v = (3*(4*a*c**3 - b**2*c**2-18*a*b*c*d+27*a**2*d**2+4*b**3*d) / (18**2*a**4)) ** (1.0/2)
	if abs(u+v) >= abs(u-v):
		m = (u+v) ** (1.0/3)
	else:
		m = (u-v) ** (1.0/3)
	if m == 0:
		n == 0
	else:
		n = (b**2-3*a*c) / (9*a**2*m)
	# w = complex(0, -0.5+(3/4)**(1.0/2))
	# w2 = complex(0, -0.5-(3/4)**(1.0/2))
	w = -0.5+(-3/4)**(1.0/2)
	w2 = -0.5-(-3/4)**(1.0/2)
	ab = -b/float(3*a)
	x1 = m+n+ab
	x2 = w*m+w2*n+ab
	x3 = w2*m+w*n+ab
	# return x1, x2, x3
	return round_ri(x1), round_ri(x2), round_ri(x3)
def solve_cubic(a, b, c, d):
	p = (3*a*c - b**2) / (3*a**2)
	q = (2*b**3 - 9*a*b*c + 27*a**2*d) / (27*a**3)
	# 判别式
	delta = (q/2)**2 + (p/3)**3
	if delta > 0:
		u = cmath.exp(cmath.log(-q/2 + cmath.sqrt(delta))/3)
		v = cmath.exp(cmath.log(-q/2 - cmath.sqrt(delta))/3)
		roots = [u + v - b/(3*a)]
	elif delta == 0:
		u = (-q/2)**(1/3)
		roots = [2*u - b/(3*a), -u - b/(3*a)]
	else:
		u = cmath.exp(cmath.log(-q/2 + cmath.sqrt(delta))/3)
		v = cmath.exp(cmath.log(-q/2 - cmath.sqrt(delta))/3)
		roots = [u + v - b/(3*a), -(u+v)/2 - b/(3*a) + cmath.sqrt(3)*(u-v)/2*1j, -(u+v)/2 - b/(3*a) - cmath.sqrt(3)*(u-v)/2*1j]
	return roots
def getTHETA(p,V,T,T_c,p_c,omega): # θ应为正值
	alpha = ALPHA(omega,T,T_c) # α(T)
	# 取 ax^3+bx^2+cx+d=0
	# 计算a -> 三次方
	A = 0.42748*alpha*0.08664*(R**3*T_c**3)/(p_c**2)
	# 计算b -> 二次方
	B = -0.42748*alpha*(R**2*T_c**2*V)/(p_c)+0.08664**2*p*V*(R**2*T_c**2)/(p_c**2)+0.08664*R*T_c*R*T*V/p_c
	# 计算c -> 一次项
	C = R*T*V*V
	# 计算d -> 常数项
	D = -p*V**3
	# 求解方程 θ^3+pθ+q=0
	P = (3*A*C-B**2)/(3*A*A)
	Q = (27*A*A*D-9*A*B*C+2*B**3)/(27*A**3)
	# Theta = np.array(SOLVE_THREE(P=P,Q=Q))-B/(3*A)
	# return Theta
	Theta = np.roots([A,B,C,D])
	solve_list = []
	for num in Theta:
		if num.imag == 0: # 实数
			solve_list.append(num.real)
	if len(solve_list) >= 2:
		cp = list(np.abs(np.array(solve_list)-1.0))
		index = cp.index(min(cp))
		return solve_list[index]
	else:
		return solve_list[0]

# 求解方程的函数
def solve_equal():
	return
# 计算V的函数
def getV(p,T,T_c,p_c,omega,theta)->list:
	# 计算a,b
	A =  0.42748*ALPHA(omega,T,T_c)*(R**2*T_c**2*theta**2)/(p_c)
	B = 0.08664*(theta)*R*T_c/(p_c)
	V = np.roots([p,-theta*R*T,-p*B*B-theta*R*T*B+A,-A*B])
	solveV = []
	for num in V:
		if num.imag == 0: # 实数
			if num.real > 0: # 体积应当为正数
				solveV.append(num.real)
	return solveV

# 需要更换来迭代求解(不需迭代)[迭代求解来得到θ(模型组任务)]
def getP(V,T,T_c,p_c,omega,theta): # 计算p的函数
	alpha = ALPHA(omega,T,T_c)
	a = 0.42748*theta**2*R**2*T_c**2*alpha/p_c
	b = 0.08664*theta*R*T_c/p_c
	return (theta*R*T)/(V-b)-a/(V*(V+b))

# 求解T的函数(修改为迭代)
def getT(p,V,T_c,p_c,omega,theta): # 计算T的函数 # 注意这里是可以液化的V,p
	# 注：θ与T无关，其中α函数含有T
	K = 0.42748*theta**2*R**2*T_c**2/p_c
	b = 0.08664*theta*R*T_c/p_c
	S = K/(V*(V+b))
	D = theta*R*T_c/(V-b)
	Rd = 0.48+1.574*omega-0.176*omega*omega
	A = D-Rd*Rd*S
	B = 2*Rd*S+2*Rd*Rd*S
	C = -S-2*Rd*S-Rd*Rd*S-p
	DELTA = B*B-4*A*C
	T1,T2 = (((-B+sqrt(DELTA))/(2*A))**2)*T_c,(((-B-sqrt(DELTA))/(2*A))**2)*T_c
	if T1 > 0 and T2 < 0:
		return T1
	elif T1 < 0 and T2 > 0:
		return T2
	elif T1 > 0 and T2 > 0:
		# 使用修正的液相来获得T
		Tmin = min([T1,T2])
		Tmax = max([T1,T2])
		return Tmin
def getThetaT(): # 后续补充的求解theta的函数
	return 0
def getTsrk(p,V,T_c,p_c,omega): # SRK方程求解T
	K = 0.42748**2*R**2*T_c**2/p_c
	b = 0.08664*R*T_c/p_c
	S = K/(V*(V+b)) # 大于0
	D = R*T_c/(V-b)
	Rd = 0.48+1.574*omega-0.176*omega*omega # 大于0
	A = D-Rd*Rd*S
	B = 2*Rd*S+2*Rd*Rd*S # 大于0
	C = -S-2*Rd*S-Rd*Rd*S-p # 小于0
	DELTA = B*B-4*A*C
	T1,T2 = (((-B+sqrt(DELTA))/(2*A))**2)*T_c,(((-B-sqrt(DELTA))/(2*A))**2)*T_c
	if T1 > 0 and T2 < 0:
		return T1
	elif T1 < 0 and T2 > 0:
		return T2
	elif T1 > 0 and T2 > 0:
		# 使用修正的液相来获得T
		Tmin = min([T1,T2])
		Tmax = max([T1,T2])
		return Tmax
# 计算逸度系数φ的函数
def Calculate_Phi(theta,p,V,T,T_c,p_c,omega,psat_list:list): # 根据拟合公式计算出的θ和带入状态方程后计算出的p,V,T
	# 同时传入计算饱和蒸气压的经验方程参数集合
	# 计算a
	A =  0.42748*ALPHA(omega,T,T_c)*(R**2*T_c**2*theta**2)/(p_c)
	B = 0.08664*(theta)*R*T_c/(p_c)

	"""为了判断是气相还是液相，需要根据已经给出的p,T,θ代入方程来反解出V"""
	V_ideal = getV(p,T,T_c,p_c,omega,theta)
	isVapor = False
	isLiquid = False
	if len(V_ideal) == 3:
		return 0
	elif len(V_ideal) == 2:
		# 分辨两个液相和气相体积
		if abs(V-min(V_ideal))<=1e-1:
			isLiquid = True
		elif abs(V-max(V_ideal))<=1e-1:
			isVapor = True
	elif len(V_ideal) == 1:
		isVapor = True # 超过临界点仅为气体
	if isVapor and not isLiquid: # 气体->直接使用【SRK方程】来得到
		# p = getP(V,T,T_c,p_c,omega,theta) 通过解方程的形式直接得到p,V,T代入，任意已知两个求第三个
		Z = p*V/(R*T)
		lnPHI = Z-1-log(Z)-(A/(B*R*T))*log(1+B/V)+log(V/((V-B)**theta))
		return exp(lnPHI)
	elif isLiquid and not isVapor:
		# 需要根据两个V来积分
		c1,c2,c3,c4,c5,c6,c7,c8,c9=psat_list[0],psat_list[1],psat_list[2],psat_list[3],psat_list[4],psat_list[5],psat_list[6],psat_list[7],psat_list[8]
		assert c8<=c9
		lnpsat = c1+c2/(T+c3)+c4*T+c5*log(T)+c6*T**(c7)
		psat = exp(lnpsat) # T下的饱和蒸汽压Pa

		# 第一部分的积分：T,p0->T,psat，需要计算出在psat下的气体Vsat
		theta = getThetaT() # psat,T下的θ(如果是这样的函数的话，就不需要变化)
		Vsatlist = getV(psat,T,T_c,p_c,omega,theta)
		Vsat = max(Vsatlist)
		Vsatl = min(Vsatlist) # 液相体积
		Zsat = psat*Vsat/(R*T)
		Asat =  0.42748*ALPHA(omega,T,T_c)*(R**2*T_c**2*theta**2)/(p_c)
		Bsat = 0.08664*(theta)*R*T_c/(p_c)
		lnPHIsat = Zsat-1-log(Zsat)-(Asat/(Bsat*R*T))*log(1+Bsat/Vsat)+log(Vsat/((Vsat-Bsat)**theta))

		# 第二部分积分：求解T,psat->T,p，注意是液相体积积分
		lnPHIliquid = (p*V-psat*Vsatl)/(R*T)-log((p*V)/(psat*Vsatl))-(Asat/(Bsat*R*T))*log(1+Bsat/V)+log(V/((V-Bsat)**theta))+log(1/((Vsatl)**(1-theta)))
		lnPHI = lnPHIsat+lnPHIliquid
		return exp(lnPHI)

def calculatePHI_VAPOR_SRK(p,V,T,Tc,pc,omega):
	"""
	使用SRK方程的形式来计算气相逸度
	"""
	asrk = A_srk(T,Tc,pc,omega)
	bsrk = B_srk(Tc,pc)
	Z = p*V/(R*T)
	lnRHI_VAPOR = Z-1-log(Z*(1-bsrk/V))-(asrk/(bsrk*R*T))*log(1+bsrk/V)
	return exp(lnRHI_VAPOR)
def calculatePHI_LIQUID_NONESRK(p,V,T,Tc,pc,omega,theta,psat_list:list)->list:
	# 假设已经是液相的V
	# 需要根据两个V来积分
	# 同时也返回此时气相逸度的计算
	Vlist = getV(p,T,Tc,pc,omega,1) # 计算气相
	V_vapor = max(Vlist)
	lnPHIvap = calculatePHI_VAPOR_SRK(p,V_vapor,T,Tc,pc,omega)

	c1,c2,c3,c4,c5,c6,c7,c8,c9=psat_list[0],psat_list[1],psat_list[2],psat_list[3],psat_list[4],psat_list[5],psat_list[6],psat_list[7],psat_list[8]
	assert c8<=c9
	lnpsat = c1+c2/(T+c3)+c4*T+c5*log(T)+c6*T**(c7)
	psat = exp(lnpsat) # T下的饱和蒸汽压Pa

	# 第一部分的积分：T,p0->T,psat，需要计算出在psat下的气体Vsat注意是气体积分
	# Vsatlist = getV(psat,T,Tc,pc,omega,theta) # 计算出在past下的体积->添加修正->使用液相V，认为其不随p变化
	Vsatlist_srk = getV(psat,T,Tc,pc,omega,1) # 用SRK方程计算
	Vsat = max(Vsatlist_srk) # 气相体积
	# Vsatl = min(Vsatlist) # 液相体积
	Vsatl = V
	lnPHIsat = calculatePHI_VAPOR_SRK(psat,Vsat,T,Tc,pc,omega) # 气相部分使用SRK

	Zsat = psat*Vsat/(R*T)
	Asat =  0.42748*ALPHA(omega,T,Tc)*(R**2*Tc**2*theta**2)/(pc)
	Bsat = 0.08664*(theta)*R*Tc/(pc)
	# 第二部分积分：求解T,psat->T,p，注意是液相体积积分
	lnPHIliquid = (p*V-psat*Vsatl)/(R*T)-log((p*V)/(psat*Vsatl))+log(V/Vsatl)-theta*log((V-Bsat)/(Vsatl-Bsat))+(Asat/(Bsat*R*T))*(log(V/Vsatl)-log((V+Bsat)/(Vsatl+Bsat)))
	lnPHI = lnPHIsat+lnPHIliquid
	return exp(lnPHI)
"""
完全的计算流程
根据输入的p,V,T来判断是什么状态下的逸度系数。(同时需要输入omega/Tc/pc,以及计算出的θ)
①我们是计算液相的逸度系数
②θ是物性和T的参数
③对于输入的p,V,T,θ,w,Tc,pc,首先需要assert【T<Tc】,这样才可以液化。否则直接将p,V,T代入气相的逸度系数求解方程即可。
②如果【T<Tc】,则需要计算液相的和气相的逸度。使用SRK方程来根据p,T计算出V,判断现在输入的V是否接近。注意有气体θ→1。如果数量级相近则分别计算气液两相。
"""

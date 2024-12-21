import pandas as pd
import numpy as np
from math import sqrt

def read_excel(filepath,kind,numc):
	excel_file = pd.ExcelFile(filepath)
	active_sheet = excel_file.sheet_names[numc]
	df = pd.read_excel(filepath,sheet_name=active_sheet)
	headers = df.columns.tolist()
	if kind == 1: # 列
		columns_data = {}
		for header in headers:
			columns_data[header] = df[header].tolist()
		return columns_data
	elif kind == 0: # 行
		rows_data = []
		for index, row in df.iterrows():
			rows_data.append(row.tolist())
		columns_data = {}
		columns_data[headers[0]] = headers[1:]
		for lists in rows_data:
			columns_data[lists[0]] = lists[1:]
		return columns_data
	
R = 8.314
def ALPHA(omega,T,T_c):
	"""
	alpha(T)函数，传入偏心因子，对比温度
	"""
	T_r = T/T_c
	return (1+(0.48+1.574*omega-0.176*omega*omega)*(1-sqrt(T_r)))**2
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
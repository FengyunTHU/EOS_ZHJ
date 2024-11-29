# EOS
化工热力学第一次大作业
基于机器学习视角修正EOS
请大家及时上传项目文件，我们可以在这里实时跟进进度！

思路一：

数据来源（data resources）：实际气体数据和高精度方程导出的PVT数据

优化对象：SRK方程（暂定）

原理：对于cubic EOS来说，每一个方程会对应一组a、b=f(Tc,Pc)，a,b表达式的系数是固定的（对于一个特定的方程）
对于不同的实际气体来说，系数不一定是固定的

优化方案：利用高精度的PVT计算出a或者b，拟合a或者b对于“分子”的曲线

### 思路——by ZHJ & WJJ：
> 引入物性参数$\theta$，改写方程如下：

$$p=\frac{\theta RT}{V-b}-\frac{a}{V(V+b)}$$其中$$a(T)=0.42748\frac{\theta^2R^2T_{c}^2}{p_c}\alpha(T),b(T)=0.08664\frac{\theta RT_c}{p_c}(\beta(T))=1$$初步将$\theta$训练为物性参数的函数，后续数据集充分的情况下训练：
$$\frac{\theta_{T=T_c}}{Z_c/0.3333}=f(\mathrm{matter}),\frac{\theta}{\theta_{T=T_c}}=g(T,\mathrm{matter})$$
代入后完整的方程为：
$$p=\frac{\theta RT}{V-0.08664\frac{\theta RT_c}{p_c}}-\frac{0.42748\frac{\theta^2R^2T_c^2}{p_c}\alpha(T)}{V(V+0.08664\frac{\theta RT_c}{p_c})}$$
其中$$\alpha(T)=[1+(0.48+1.574\omega-0.176\omega^2)(1-(\frac{T}{T_c})^{0.5})]^2$$
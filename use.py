import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo

NAME_LIST = ["112-42-5","108-93-0","107-18-6","77-99-6","109-52-4","64-19-7","638-49-3","110-19-0"]
def read_excel(filepath, kind):
    excel_file = pd.ExcelFile(filepath)
    active_sheet = excel_file.sheet_names[0]
    df = pd.read_excel(filepath, sheet_name=active_sheet)
    headers = df.columns.tolist()
    if kind == 1:  # 列
        columns_data = {}
        for header in headers:
            columns_data[header] = df[header].tolist()
        return columns_data
    elif kind == 0:  # 行
        rows_data = []
        for index, row in df.iterrows():
            rows_data.append(row.tolist())
        columns_data = {}
        columns_data[headers[0]] = headers[1:]
        for lists in rows_data:
            columns_data[lists[0]] = lists[1:]
        return columns_data

df = read_excel(f'./HANDLE_data/ThetaDataAcid(4种acid,可用Zc分辨边界t).xlsx', 1)
Theta_DataSet = df['θ']
T_r_DataSet = df['Tr']
p_r_DataSet = df['pr']
print(len(Theta_DataSet))

# 定义切片范围和颜色3338
slices = [(0,3338),(5037,6196),(6196,7553)]
colors = ['orange','pink','blue']

# 创建三维图
traces = []
for i, (start, end) in enumerate(slices):
    trace = go.Scatter3d(
        x=T_r_DataSet[start:end],
        y=p_r_DataSet[start:end],
        z=Theta_DataSet[start:end],
        mode='markers',
        marker=dict(
            size=1,
            color=colors[i]
        ),
        name=f'Slice {i+1}'
    )
    traces.append(trace)

layout = go.Layout(
    title='3D Scatter Plot',
    scene=dict(
        xaxis=dict(title='Tr'),
        yaxis=dict(title='pr'),
        zaxis=dict(title='θ')
    )
)

fig = go.Figure(data=traces, layout=layout)

# 显示图形
pyo.iplot(fig)
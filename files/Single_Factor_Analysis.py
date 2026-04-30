#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import warnings

warnings.filterwarnings("ignore")
os.makedirs("files/report_images", exist_ok=True)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import statsmodels
import yfinance as yf
import openpyxl


# In[2]:


# ===== 生成可跑通的模拟 Excel 数据：不用下载，直接在本地生成 =====

import sys
import subprocess
import importlib.util
import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ===== 基础设置 =====
dates = pd.bdate_range("2023-01-02", periods=120)

# 生成 50 只模拟股票代码
stocks = [f"600{i:03d}.SH" for i in range(50)]

# ===== 1. 上证50指增sig.xlsx =====

# 是否在股票池内：全部为 1
df_50sig = pd.DataFrame(1, index=dates, columns=stocks)
df_50sig.index.name = "date"

# 停牌信号：大多数为 0，少量为 1
df_stop_sig = pd.DataFrame(
    np.random.choice([0, 1], size=(len(dates), len(stocks)), p=[0.97, 0.03]),
    index=dates,
    columns=stocks
)
df_stop_sig.index.name = "date"

# 涨跌停信号：大多数为 0，少量为 1
df_limit_sig = pd.DataFrame(
    np.random.choice([0, 1], size=(len(dates), len(stocks)), p=[0.95, 0.05]),
    index=dates,
    columns=stocks
)
df_limit_sig.index.name = "date"

# 日收益率：模拟正态分布收益
df_ret = pd.DataFrame(
    np.random.normal(0.0005, 0.02, size=(len(dates), len(stocks))),
    index=dates,
    columns=stocks
)
df_ret.index.name = "date"

with pd.ExcelWriter("上证50指增sig.xlsx", engine="openpyxl") as writer:
    df_50sig.to_excel(writer, sheet_name="50sig")
    df_stop_sig.to_excel(writer, sheet_name="stop_sig")
    df_limit_sig.to_excel(writer, sheet_name="limit_sig")
    df_ret.to_excel(writer, sheet_name="ret")


# ===== 2. 上证50指增数据.xlsx =====

def make_factor(mean, std, positive=False):
    data = np.random.normal(mean, std, size=(len(dates), len(stocks)))
    if positive:
        data = np.abs(data) + 0.01
    df = pd.DataFrame(data, index=dates, columns=stocks)
    df.index.name = "date"
    return df

# 自由流通市值，必须为正，因为后面会取 log
df_free_value = make_factor(1e10, 2e9, positive=True)

# 市净率 PB
df_pb = make_factor(1.8, 0.5, positive=True)

# 换手率
df_turnover_rate = make_factor(0.03, 0.01, positive=True)# 一、数据导入
sig_file = pd.ExcelFile("上证50指增sig.xlsx")
factor_file = pd.ExcelFile("上证50指增数据.xlsx")
ind_file = pd.ExcelFile("上证50行业.xlsx")

sig_names = sig_file.sheet_names
factor_names = factor_file.sheet_names

df_50sig = pd.read_excel(sig_file, sig_names[0], index_col='date')
df_stop_sig = pd.read_excel(sig_file, sig_names[1], index_col='date')
df_limit_sig = pd.read_excel(sig_file, sig_names[2], index_col='date')
df_ret = pd.read_excel(sig_file, sig_names[3], index_col='date')

df_free_value = pd.read_excel(factor_file, factor_names[0], index_col='date')
df_pb = pd.read_excel(factor_file, factor_names[1], index_col='date')
df_turnover_rate = pd.read_excel(factor_file, factor_names[2], index_col='date')
df_mom = pd.read_excel(factor_file, factor_names[3], index_col='date')
df_std = pd.read_excel(factor_file, factor_names[4], index_col='date')
df_roe = pd.read_excel(factor_file, factor_names[5], index_col='date')
df_beta = pd.read_excel(factor_file, factor_names[6], index_col='date')
df_dy = pd.read_excel(factor_file, factor_names[7], index_col='date')
df_free_value = df_free_value.astype(float)

# 先把 0 和负数变成 NaN，避免 log(0) 或 log(负数)
df_free_value = df_free_value.where(df_free_value > 0)

# 再取 log
df_free_value = np.log(df_free_value)

# 最后把 NaN / inf 清掉
df_free_value = df_free_value.replace([np.inf, -np.inf], np.nan).fillna(0)

ind = pd.read_excel(ind_file, ind_file.sheet_names[0], index_col='date')
ind_names = pd.read_excel(ind_file, ind_file.sheet_names[1])

print("数据导入成功")
print(df_ret.head())

# 动量
df_mom = make_factor(0.02, 0.08, positive=False)

# 波动率
df_std = make_factor(0.25, 0.05, positive=True)

# ROE
df_roe = make_factor(0.12, 0.04, positive=False)

# Beta
df_beta = make_factor(1.0, 0.2, positive=False)

# 股息率
df_dy = make_factor(0.03, 0.01, positive=True)

with pd.ExcelWriter("上证50指增数据.xlsx", engine="openpyxl") as writer:
    df_free_value.to_excel(writer, sheet_name="free_value")
    df_pb.to_excel(writer, sheet_name="pb")
    df_turnover_rate.to_excel(writer, sheet_name="turnover_rate")
    df_mom.to_excel(writer, sheet_name="mom")
    df_std.to_excel(writer, sheet_name="std")
    df_roe.to_excel(writer, sheet_name="roe")
    df_beta.to_excel(writer, sheet_name="beta")
    df_dy.to_excel(writer, sheet_name="dy")


# ===== 3. 上证50行业.xlsx =====

industry_codes = [1, 2, 3, 4, 5]
industry_names = ["Finance", "Energy", "Consumer", "Technology", "Healthcare"]

# 每只股票固定一个行业
stock_industry = {
    stock: np.random.choice(industry_codes)
    for stock in stocks
}

ind = pd.DataFrame(index=dates, columns=stocks)
for stock in stocks:
    ind[stock] = stock_industry[stock]
ind.index.name = "date"

ind_names = pd.DataFrame({
    "industry_code": industry_codes,
    "industry_name": industry_names
})

with pd.ExcelWriter("上证50行业.xlsx", engine="openpyxl") as writer:
    ind.to_excel(writer, sheet_name="industry")
    ind_names.to_excel(writer, sheet_name="industry_names", index=False)


# ===== 生成后立刻验证 =====

print("文件生成完成：")
for file in ["上证50指增sig.xlsx", "上证50指增数据.xlsx", "上证50行业.xlsx"]:
    print(file, "存在" if os.path.exists(file) else "不存在")

print("\n检查 df_50sig 前 5 行：")
print(df_50sig.head())

print("\n检查 df_ret 前 5 行：")
print(df_ret.head())

print("\n检查 df_free_value 前 5 行：")
print(df_free_value.head())

print("\n检查是否有 NaN：")
print("df_50sig NaN 数量：", df_50sig.isna().sum().sum())
print("df_ret NaN 数量：", df_ret.isna().sum().sum())
print("df_free_value NaN 数量：", df_free_value.isna().sum().sum())


# In[ ]:





# In[ ]:





# In[3]:


# 二、数据初步筛选

# 选择因子
raw_factor = df_turnover_rate.copy()

# 如果 stop_sig / limit_sig 中：
# 0 = 正常可交易
# 1 = 停牌或涨跌停，不能交易
tradable_mask = (
    (df_50sig == 1) &
    (df_stop_sig == 0) &
    (df_limit_sig == 0)
)

df_factor = raw_factor.where(tradable_mask)

# ret 提前一阶，方便计算下一期收益
df_ret = df_ret.shift(-1).fillna(0)

print("筛选后每个交易日有效股票数量：")
print(df_factor.notna().sum(axis=1).describe())

print("df_factor 是否全是 NaN：", df_factor.isna().all().all())
print(df_factor.head())


# In[4]:


#三、极端值剔除————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————N倍MAD剔除法（如果是全市场选股的话，这个剔除要放在筛选的前面）——————————————————————————————————————————————————
# 计算每行的均值和标准差
row_med = df_factor.median(axis=1)
row_mad = 1.4826 * abs(df_factor-row_med.to_numpy().reshape(-1, 1)).median(axis=1)
# 计算剔除条件
lower_bound = row_med - 3 * row_mad
upper_bound = row_med + 3 * row_mad
lower_bound =  lower_bound.to_numpy().reshape(-1, 1)
upper_bound =  upper_bound.to_numpy().reshape(-1, 1)
# 将超出范围的值设置为 NaN
df_factor[(df_factor < lower_bound) | (df_factor > upper_bound)] = np.nan
#————————————————————————————————————————————N倍标准差剔除法（如果是全市场选股的话，这个剔除要放在筛选的前面）——————————————————————————————————————————————————
# 计算每行的均值和标准差
#row_mean = df_factor.mean(axis=1)
#row_std = df_factor.std(axis=1)
# 计算剔除条件
#lower_bound = row_mean - 3 * row_std
#upper_bound = row_mean + 3 * row_std
#lower_bound =  lower_bound.to_numpy().reshape(-1, 1)
#upper_bound =  upper_bound.to_numpy().reshape(-1, 1)
# 将超出范围的值设置为 NaN
#df_factor[(df_factor < lower_bound) | (df_factor > upper_bound)] = np.nan
#————————————————————————————————————————————分位数剔除极值（如果是全市场选股的话，这个剔除要放在筛选的前面）——————————————————————————————————————————————————
#lower_bound = df_factor.quantile(0.01, axis=1)
#upper_bound  = df_factor.quantile(0.99, axis=1)
#lower_bound =  lower_bound.to_numpy().reshape(-1, 1)
#upper_bound =  upper_bound.to_numpy().reshape(-1, 1)
# 将超出范围的值设置为 NaN
#df_factor[(df_factor < lower_bound) | (df_factor > upper_bound)] = np.nan


# In[5]:


# 四、市值、行业中性化

def neutralize_one_day(factor_row):
    date = factor_row.name

    y = factor_row.copy()
    size_row = df_free_value.loc[date].reindex(y.index)
    ind_row = ind.loc[date].reindex(y.index)

    valid = y.notna() & size_row.notna() & ind_row.notna()

    # 有效股票太少时，不做回归
    if valid.sum() < 5:
        return pd.Series(np.nan, index=y.index)

    y_valid = y[valid].astype(float)
    size_valid = size_row[valid].astype(float)
    ind_valid = ind_row[valid]

    # 行业 dummy，自动删除一个基准行业，避免完全共线
    ind_dummies = pd.get_dummies(
        ind_valid.astype(str),
        prefix="ind",
        drop_first=True,
        dtype=float
    )

    X = pd.DataFrame({
        "const": 1.0,
        "size": size_valid
    }, index=y_valid.index)

    X = pd.concat([X, ind_dummies], axis=1)

    # 样本数必须大于解释变量数量
    if len(y_valid) <= X.shape[1]:
        return pd.Series(np.nan, index=y.index)

    beta = np.linalg.lstsq(X.values, y_valid.values, rcond=None)[0]
    resid = y_valid.values - X.values @ beta

    out = pd.Series(np.nan, index=y.index)
    out.loc[y_valid.index] = resid

    return out

df_factor = df_factor.apply(neutralize_one_day, axis=1).round(5)

print("中性化后每个交易日有效股票数量：")
print(df_factor.notna().sum(axis=1).describe())

print("中性化后 df_factor 是否全是 NaN：", df_factor.isna().all().all())
print(df_factor.head())


# In[6]:


#五、因子标准化————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————Z_score标准化——————————————————————————————————————————————————————————————————————————
#df_factor = df_factor.apply(lambda x: (x-x.mean())/(x.std()), axis=1)
#————————————————————————————————————————————————0-1标准化——————————————————————————————————————————————————————————————————————————
df_factor = df_factor.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)

file_path = "单因子集.xlsx"

# # 使用 ExcelWriter 写入文件
# with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
#     # 假设你想将数据框 df 写入新的 sheet，名称为 '新Sheet'
#     df_factor.to_excel(writer, sheet_name='股息率', index=True)  # 设置 index=True


# In[7]:


#六、分组测试————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 计算每一行的剔除nan以后，从大到小分5组，每组的分界线（排序等分法）
g=5
g_p = [i / g for i in range(1, g)]
thresholds_factor = df_factor.apply(lambda row: list(row.dropna().nlargest(int(len(row.dropna()) * p)).min() for p in g_p), axis=1)
# 计算每一行的剔除nan以后，从大到小分5组，每组的分界线（分位数等分法，可以用，但是不常用，相当于在分组的基础上引入了因子本身）
#thresholds_factor = df_factor.apply(lambda row: list(row.dropna().quantile([0.2, 0.4, 0.6, 0.8])), axis=1)
# 根据阈值进行标记
factor_group = df_factor.apply(lambda row: row.apply(lambda x: sum(x >= t for t in thresholds_factor[row.name]) + 1 if pd.notnull(x) else np.nan), axis=1)
#分组计算收益率和净值
#ret为了方便要跟着df_factor来剔除
ret_factor=df_factor.notna().astype(int)*df_ret
ret_factor.replace(0, np.nan, inplace=True)
ret = []
jz= []

#每一组组内等权
for i in range(1, g+1):
    ret_i = (factor_group == i) * ret_factor
    ret_i.replace(0, np.nan, inplace=True)
    ret_i=ret_i.mean(axis=1)
    ret_i.fillna(0,inplace=True)
    ret.append(ret_i)
    # 将每个收益率加1，得到增长率
    growth_rates = 1 + ret_i
    jz_i= np.cumprod(growth_rates)
    jz.append(jz_i)


# In[8]:


ret_rank = ret_factor.rank(axis=1)
factor_rank = df_factor.rank(axis=1)

ic = df_factor.corrwith(ret_factor, axis=1)
rankic = factor_rank.corrwith(ret_rank, axis=1)

icir = ic.rolling(12).mean() / ic.rolling(12).std()
rankicir = rankic.rolling(12).mean() / rankic.rolling(12).std()

ic_cumsum = ic.cumsum()
icir_cumsum = icir.cumsum()
rankic_cumsum = rankic.cumsum()
rankicir_cumsum = rankicir.cumsum()


def plot_dual_axis_bar_line(df1, df2, ylabel1, ylabel2, title, filename):
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200)

    x = df1.index.strftime('%Y-%m-%d')

    ax1.plot(x, df1, marker='o', linestyle='-', color='b', markersize=4)
    ax1.set_xlabel('Date')
    ax1.set_ylabel(ylabel1, color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(x, df2, marker='o', linestyle='-', color='r', markersize=4)
    ax2.set_ylabel(ylabel2, color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title(title)
    plt.xticks(rotation=45)

    fig.tight_layout()
    fig.savefig(f"files/report_images/{filename}", bbox_inches="tight")

    plt.show()
    plt.close(fig)


plot_dual_axis_bar_line(
    ic_cumsum,
    icir_cumsum,
    'IC-cumsum',
    'ICIR-cumsum',
    'IC and ICIR-cumsum',
    'ic_icir_cumsum.png'
)

plot_dual_axis_bar_line(
    rankic_cumsum,
    rankicir_cumsum,
    'Rank IC-cumsum',
    'Rank ICIR-cumsum',
    'Rank IC and Rank ICIR-cumsum',
    'rankic_rankicir_cumsum.png'
)


# In[9]:


# ————————————————————————————————————净值画图并保存————————————————————————————————————————————————————————

fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

group_num = 0
for jz_i in jz:
    group_num += 1
    ax.plot(
        jz_i.index,
        jz_i.values,
        marker='o',
        linestyle='-',
        label=f'G{group_num}',
        markersize=4
    )

ax.plot(jz[-1] / jz[0], label='max/min')

ax.set_xlabel('date')
ax.set_ylabel('net_value')
ax.set_title('Grouped Net Value Performance')
ax.legend()

fig.tight_layout()
fig.savefig("files/report_images/group_net_value.png", bbox_inches="tight")
plt.show()
plt.close(fig)


# In[10]:


print("factor shape:", df_factor.shape)
print("ret shape:", df_ret.shape)

print("factor date range:", df_factor.index.min(), "to", df_factor.index.max())
print("ret date range:", df_ret.index.min(), "to", df_ret.index.max())

print("average valid stocks per date:")
print(df_factor.notna().sum(axis=1).describe())


# In[11]:


"df_factor" in globals()


# In[12]:


for name in ["df_ret", "df_factor", "ret", "jz", "ic", "rankic"]:
    print(name, name in globals())


# In[13]:


html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Single Factor Analysis Report</title>
  <link rel="stylesheet" href="../style.css">
  <style>
    .report-container {
      max-width: 1000px;
      margin: 50px auto;
      padding: 20px;
      text-align: center;
    }

    .report-section {
      margin: 40px 0;
      padding: 25px;
      background: white;
      border-radius: 16px;
      box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }

    .report-section img {
      width: 100%;
      max-width: 900px;
      border-radius: 12px;
      border: 1px solid #ddd;
    }

    .report-section h2 {
      color: #2193b0;
    }

    .report-section p {
      color: #444;
      line-height: 1.7;
    }
  </style>
</head>

<body>
  <div class="report-container">
    <h1>AI-Assisted A-Share Single Factor Analysis</h1>
    <p>
      This report presents the key visual results of a single-factor analysis workflow,
      including IC performance, Rank IC performance, and grouped net value comparison.
    </p>

    <div class="report-section">
      <h2>IC and ICIR Cumulative Performance</h2>
      <img src="report_images/ic_icir_cumsum.png" alt="IC and ICIR cumulative performance chart">
    </div>

    <div class="report-section">
      <h2>Rank IC and Rank ICIR Cumulative Performance</h2>
      <img src="report_images/rankic_rankicir_cumsum.png" alt="Rank IC and Rank ICIR cumulative performance chart">
    </div>

    <div class="report-section">
      <h2>Grouped Net Value Performance</h2>
      <img src="report_images/group_net_value.png" alt="Grouped net value performance chart">
    </div>
  </div>
</body>
</html>
"""

with open("files/Single_Factor_Analysis_Report.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Clean HTML report generated.")


# In[ ]:


get_ipython().system('jupyter nbconvert --to script Single_Factor_Analysis-Copy1.ipynb')


# In[ ]:





# In[ ]:





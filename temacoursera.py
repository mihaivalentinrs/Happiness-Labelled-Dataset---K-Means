import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv('dataset.csv')

print("\nThe data set has been read.")
print("\nSelecting sets of data...")

#Selecting the gdp data and avg_income of individuals

gdp = data['GDP']
income = data['avg_income']
print("\nThe selected gdp data: ")
print(gdp.head())
print("\nThe selected income data: ")
print(income.head())
print("\n")
#Using numpy to prepare our dataset for clustering
gdp_income = np.column_stack((gdp, income))
km_rest = KMeans(n_clusters=6).fit(gdp_income)
clusters = km_rest.cluster_centers_


plt.scatter(clusters[:,0], clusters[:,1], s=350)
#Labeling the highest rated gdp to show the connection between data sets

high_gdp = data[ data['GDP'] > 1.3 ]
plt.scatter(high_gdp['avg_income'], high_gdp['GDP'], alpha = 0.75)
plt.scatter(high_gdp.iloc[0]['avg_income'], high_gdp.iloc[0]['GDP'])
plt.scatter(high_gdp.iloc[-1]['avg_income'], high_gdp.iloc[-1]['GDP'])
for k, row in high_gdp.iterrows():
    plt.text(row['avg_income'], row['GDP'], row['country'])

plt.title("Highest GDP/INCOME correlation")
plt.xlabel("--INCOME--")
plt.ylabel("--Highest GDP--")
plt.show()
print("\nThe labeled graph has been displayed.")


#Solving the rest of the task
std_dev_inc = np.std(income, ddof=1) #ddof is delta degrees of freedom which means that this set of code executes the std dev with the bessel s correction
mean_inc = np.mean(income)
std_dev_gdp = np.std(gdp, ddof=1)
mean_gdp = np.mean(gdp)
print("\nSorting data...")
data.sort_values(by=['GDP'], ascending=False, inplace=True)
data.sort_values(by=['avg_income'], ascending=False, inplace=True)
print("\nThe data has been sorted in ascending order.")

sum_gdp_income = 0
sx = 0
sy = 0
#correl(x,y)  = cov(x,y)/sx*sy
for i in range(len(gdp)):
    sum_gdp_income = sum_gdp_income+(gdp[i] - mean_gdp)*(income[i]-mean_inc)
    sx = sx+(gdp[i] - mean_gdp)*(gdp[i] - mean_gdp)
    sy = sy+(income[i] - mean_inc)*(income[i] - mean_inc)

coef_corel = sum_gdp_income/(sx*sy)
print("\nCorrelation coefficient: ", coef_corel)
coef_corel_py = np.corrcoef(income, gdp)[0,1]


plt.scatter(gdp, income, alpha = 0.75) #transparency for the dots
plt.scatter(clusters[:,0], clusters[:,1], s=350, alpha = 0.75)
plt.title("GDP and Income CORRELATION", fontsize = 20, fontweight = 'bold')
plt.xlabel("GDP", fontsize = 15)
plt.ylabel("Income", fontsize = 15)
plt.text(0, 25000, f"GDP Mean= ${mean_gdp:.2f}" , fontsize = 10)
plt.text(0, 24000, f"Income Mean= ${mean_inc:.2f}" , fontsize = 10)
plt.text(0,23000, f"GDP Standard Deviation= ${std_dev_inc:.2f}" , fontsize = 10)
plt.text(0, 22000, f"Income Standard Deviation= ${std_dev_gdp:.2f}" , fontsize = 10)
plt.text(0, 21000, f"Correlation Coefficient= ${coef_corel_py:.2f}" , fontsize = 10)
plt.show()
print("\nThe first graph.")
#By eye-labeling the graph we can clearly see that there is a direct relationship between gdp and avg_income as the correlation is positive.

plt.scatter(gdp, income, alpha = 0.65)
plt.title("GDP and Income CORRELATION - Country Labels", fontsize = 45)
plt.xlabel("GDP", fontsize = 25, fontweight = 'bold')
plt.ylabel("Income", fontsize = 25, fontweight = 'bold')
for index, row in data.iterrows():
    plt.text(row['GDP'], row['avg_income'], row['country'])

#plt.show()

print("\nThe second graph.")
print("\nBasic statistics: ")
print("\nMean of income: ", mean_inc)
print("\nStd of income: ", std_dev_inc)
print("\nMean of gdp: ", mean_gdp)
print("\nStd of gdp: ", std_dev_gdp)

print("\nThe task has been solved and shows the actual correlation between the two sets of data selected.")
print("\n")


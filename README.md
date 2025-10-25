K-Means Python Exercise

🌍 GDP vs Average Income – Clustering and Correlation Analysis
📘 Project Overview

This project explores the relationship between GDP and average income across different countries using K-Means clustering and statistical analysis.
It aims to visualize how GDP correlates with income levels, identify clusters of economically similar countries, and compute core statistical indicators such as mean, standard deviation, and correlation coefficient.

⚙️ Steps Implemented
1. Data Loading

The dataset (dataset.csv) is imported into a Pandas DataFrame.
It contains at least the following columns:

country – name of each country

GDP – Gross Domestic Product

avg_income – average income per individual

2. Data Selection

The analysis focuses on two numeric columns:

GDP

avg_income

These are extracted and prepared as a NumPy matrix for clustering.

3. K-Means Clustering

K-Means is applied with 6 clusters to group countries by GDP and income similarity.
Cluster centroids are visualized in scatter plots, giving insight into distinct economic groupings.

4. Visualization

Several plots are created:

Cluster Plot: Shows the relationship between GDP and average income, with cluster centers highlighted.

High-GDP Label Plot: Highlights countries with the highest GDP values and labels them directly on the graph.

Correlation Plot: Displays the overall relationship between GDP and income, including calculated statistics and correlation coefficient.

Country Label Plot: Annotates every country for easy interpretation of distribution patterns.

5. Statistical Measures

Key statistical metrics are calculated:

Mean (GDP & Income)

Standard Deviation (GDP & Income) – using Bessel’s correction (ddof=1)

Correlation Coefficient (manual and NumPy-based)

These help quantify how strongly GDP and income are related.

📊 Results & Insights

Positive Correlation:
The computed correlation coefficient (close to +1) indicates a strong positive relationship between GDP and average income.
As GDP increases, average income generally rises as well.

Cluster Insights:
The K-Means clustering identifies 6 economic groups, separating low-income/low-GDP countries from high-income/high-GDP ones.

Statistical Summary:
The means and standard deviations show significant economic variance among the countries in the dataset.

🧠 Evaluation

This dataset is well-suited for clustering and correlation analysis:

The numerical data (GDP and income) allows meaningful Euclidean distance computations.

K-Means effectively separates regions based on economic scale.

However, since GDP and income are linearly correlated, the clusters may align along a trend line rather than forming distinct circular groups.

The project provides a clear visualization of global economic disparities and can serve as a foundation for deeper studies (e.g., regression modeling, PCA, or socioeconomic forecasting).

📦 Technologies Used

Python

NumPy

Pandas

Matplotlib

scikit-learn (KMeans)

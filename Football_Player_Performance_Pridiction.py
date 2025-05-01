---
#Data Preparation
---

###Import Packages
"""

import numpy as np
import pandas as pd
from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, silhouette_samples, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

"""### Dataset Exploration
Dataset(features): https://www.kaggle.com/datasets/vivovinco/20212022-football-player-stats?select=2021-2022+Football+Player+Stats.csv

Dataset(labels): https://www.kaggle.com/datasets/vivovinco/20222023-football-player-stats
Task: perform linear regression to predict "Sales"
"""

drive.mount('/content/drive')
features_df = pd.read_csv('/content/drive/My Drive/COMP4211/Proposal/2122_data.csv', encoding='latin1', delimiter=';')
labels_df = pd.read_csv('/content/drive/My Drive/COMP4211/Proposal/2223_data.csv', encoding='latin1', delimiter=';')

labels_df['Assists'] = (labels_df['Assists'] * labels_df['MP']).round().astype(int)
data = pd.merge(features_df, labels_df[['Player', 'Goals', 'Assists']], on='Player', how='inner')

data

data.info()

data.describe()

numerical_data = data.select_dtypes(include=['int64', 'float64'])


plt.figure(figsize=(100, 60))
sns.heatmap(numerical_data.corr(), annot=True, cmap='YlGnBu', linecolor='r', linewidths=0.5)
plt.title('Heat Map')
plt.show()

"""## Preprocessing

###Identifying types of features
"""

categorical_data = data.select_dtypes(include=["object"])
print(f"Categorical features are: {categorical_data.columns.tolist()}")
print(f"\n{len(categorical_data.columns)} categorical features in total.")

continuous_numerical_data = data.select_dtypes(include=["float64"])
print(f"\n\nContinuous numerical features are: {continuous_numerical_data.columns.tolist()}")
print(f"\n{len(continuous_numerical_data.columns)} continuous numerical features in total.")

discrete_numerical_data = data.select_dtypes(include=["int64"])
print(f"\n\nDiscrete numerical features are: {discrete_numerical_data.columns.tolist()}")
print(f"\n{len(discrete_numerical_data.columns)} discrete numerical features in total.")

"""### Handling Missing Value"""

def check_missing_values(df):
    missing_values = df.isnull().sum()
    missing_features = missing_values[missing_values > 0]
    print("Features with missing values:\n" , missing_features)
    print("Proportion of missing values:\n", missing_features / len(df))

check_missing_values(data)

def impute_missing_values(df, strategy='median'):
    imputer = SimpleImputer(strategy=strategy)
    imputed_data = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_data, columns=df.columns)
    return imputed_df

numerical_data_imputed = impute_missing_values(numerical_data, strategy='median')
numerical_data_imputed

"""### Standardization"""

def scale_numerical_data(df, method='standard'):
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    return scaled_df


numerical_training_data_standard = scale_numerical_data(numerical_data_imputed, method='standard')
numerical_training_data_minmax = scale_numerical_data(numerical_data_imputed, method='minmax')
numerical_training_data_robust = scale_numerical_data(numerical_data_imputed, method='robust')

print(numerical_data_imputed.head())

print("Standard Scaled Data:")
print(numerical_training_data_standard.head())

print("\nMin-Max Scaled Data:")
print(numerical_training_data_minmax.head())

print("\nRobust Scaled Data:")
print(numerical_training_data_robust.head())

"""###Encoding
---

Consider "player" as key. There is no need to do one hot encoding for "player".
"""

def preprocess_categorical_data(data):

    categorical_columns = data.select_dtypes(exclude=['number']).columns.tolist()

    encoder = OrdinalEncoder()
    encoded_data = encoder.fit_transform(data[categorical_columns])

    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

    return encoded_df

encoded_df = preprocess_categorical_data(categorical_data)
encoded_df

"""## Feature Selection"""

def select_best_features(X, y, k=23):

    selector_goal = SelectKBest(score_func=f_classif, k=k)
    selector_goal.fit(X, y.iloc[:, -2])
    scores_goal = selector_goal.scores_

    selector_assist = SelectKBest(score_func=f_classif, k=k)
    selector_assist.fit(X, y.iloc[:, -1])
    scores_assist = selector_assist.scores_

    combined_scores = scores_goal + scores_assist
    selected_indices = combined_scores.argsort()[-k:][::-1]

    score_df = pd.DataFrame({
        'Feature': X.columns,
        'Goal Score': scores_goal,
        'Assist Score': scores_assist,
        'Combined Score': combined_scores
    }).sort_values(by='Combined Score', ascending=False)

    X_selected = X.iloc[:, selected_indices]

    return score_df, X_selected, selected_indices

score_df, X_selected, selected_indices = select_best_features(
    X=numerical_training_data_robust.iloc[:, :-2],
    y=data.iloc[:, -2:],
    k=28
)

score_df[:28]

"""---
# Regression
---

##Preprocess data
"""

def preprocess(data):
    data_without_player = data.drop(columns=['Player'], errors='ignore')
    numerical_data = data_without_player.select_dtypes(include=['number'])
    categorical_data_encoded = preprocess_categorical_data(data_without_player)
    combined_data = pd.concat([categorical_data_encoded, numerical_data], axis=1)
    data_imputed = impute_missing_values(combined_data)
    scaled_data = scale_numerical_data(data_imputed, "robust")
    score_df, X_selected, selected_indices = select_best_features(
    X=scaled_data.iloc[:, :-2],
    y=data,
    k=23
    )
    selected_columns = scaled_data.columns[:-2][selected_indices]
    final_data = pd.DataFrame(X_selected, columns=selected_columns)
    final_data = pd.concat([data['Player'], final_data], axis=1)
    final_data = pd.concat([final_data, data.iloc[:, -2:]], axis=1)

    return final_data

preprocessed_data = preprocess(data)
preprocessed_data

"""### Data Split

Spliting data into x and y. Further split the data into train and test with the training-to-validation ratio 80:20. Random_state is set to 4211.
"""

x = preprocessed_data.iloc[:, 1:-2]
y = data[['Goals_y', 'Assists_y']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4211)

"""## Building the regression model

###Linear Regression
"""

model = MultiOutputRegressor(LinearRegression())
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse_goals = mean_squared_error(y_test['Goals_y'], y_pred[:, 0])
mse_assists = mean_squared_error(y_test['Assists_y'], y_pred[:, 1])
r2_goals = r2_score(y_test['Goals_y'], y_pred[:, 0])
r2_assists = r2_score(y_test['Assists_y'], y_pred[:, 1])

print(f'Goals - MSE: {mse_goals}, R^2: {r2_goals}')
print(f'Assists - MSE: {mse_assists}, R^2: {r2_assists}')

"""###Random Forest"""

model = RandomForestRegressor(n_estimators=100)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse_goals = mean_squared_error(y_test['Goals_y'], y_pred[:, 0])
mse_assists = mean_squared_error(y_test['Assists_y'], y_pred[:, 1])
r2_goals = r2_score(y_test['Goals_y'], y_pred[:, 0])
r2_assists = r2_score(y_test['Assists_y'], y_pred[:, 1])

print(f'Goals - MSE: {mse_goals}, R^2: {r2_goals}')
print(f'Assists - MSE: {mse_assists}, R^2: {r2_assists}')

"""###Deceision Tree"""

model = DecisionTreeRegressor()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse_goals = mean_squared_error(y_test['Goals_y'], y_pred[:, 0])
r2_goals = r2_score(y_test['Goals_y'], y_pred[:, 0])
mse_assists = mean_squared_error(y_test['Assists_y'], y_pred[:, 1])
r2_assists = r2_score(y_test['Assists_y'], y_pred[:, 1])

print(f'Goals - MSE: {mse_goals}, R^2: {r2_goals}')
print(f'Assists - MSE: {mse_assists}, R^2: {r2_assists}')

"""###Gradient Boosting"""

model = MultiOutputRegressor(GradientBoostingRegressor())
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse_goals = mean_squared_error(y_test['Goals_y'], y_pred[:, 0])
r2_goals = r2_score(y_test['Goals_y'], y_pred[:, 0])
mse_assists = mean_squared_error(y_test['Assists_y'], y_pred[:, 1])
r2_assists = r2_score(y_test['Assists_y'], y_pred[:, 1])

print(f'Goals - MSE: {mse_goals}, R^2: {r2_goals}')
print(f'Assists - MSE: {mse_assists}, R^2: {r2_assists}')

"""###SVR"""

model = MultiOutputRegressor(SVR(kernel='rbf'))
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse_goals = mean_squared_error(y_test['Goals_y'], y_pred[:, 0])
r2_goals = r2_score(y_test['Goals_y'], y_pred[:, 0])
mse_assists = mean_squared_error(y_test['Assists_y'], y_pred[:, 1])
r2_assists = r2_score(y_test['Assists_y'], y_pred[:, 1])

print(f'Goals - MSE: {mse_goals}, R^2: {r2_goals}')
print(f'Assists - MSE: {mse_assists}, R^2: {r2_assists}')

"""###Ridge"""

model = Ridge(alpha=2.0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse_goals = mean_squared_error(y_test['Goals_y'], y_pred[:, 0])
r2_goals = r2_score(y_test['Goals_y'], y_pred[:, 0])
mse_assists = mean_squared_error(y_test['Assists_y'], y_pred[:, 1])
r2_assists = r2_score(y_test['Assists_y'], y_pred[:, 1])

print(f'Goals - MSE: {mse_goals}, R^2: {r2_goals}')
print(f'Assists - MSE: {mse_assists}, R^2: {r2_assists}')

def select_and_sort_features(X, y, k=23):
    from sklearn.feature_selection import SelectKBest, f_classif
    import pandas as pd

    selector_goal = SelectKBest(score_func=f_classif, k=k)
    selector_goal.fit(X, y.iloc[:, -2])
    scores_goal = selector_goal.scores_

    selector_assist = SelectKBest(score_func=f_classif, k=k)
    selector_assist.fit(X, y.iloc[:, -1])
    scores_assist = selector_assist.scores_

    score_df = pd.DataFrame({
        'Feature': X.columns,
        'Goal Score': scores_goal,
        'Assist Score': scores_assist
    })

    sorted_goal_indices = selector_goal.get_support(indices=True)
    sorted_assist_indices = selector_assist.get_support(indices=True)

    X_selected_goals = X.iloc[:, sorted_goal_indices]
    X_selected_assists = X.iloc[:, sorted_assist_indices]

    return score_df, X_selected_goals, X_selected_assists

for k in range(13,31):
    print(f"\nNow k = {k}")

    score_df, X_selected_goals, X_selected_assists = select_and_sort_features(
        X=numerical_training_data_robust.iloc[:, :-2],
        y=data.iloc[:, -2:],
        k=k
    )

    X_train_goals, X_test_goals, y_train_goals, y_test_goals = train_test_split(
        X_selected_goals, data['Goals_y'], test_size=0.2, random_state=42
    )

    X_train_assists, X_test_assists, y_train_assists, y_test_assists = train_test_split(
        X_selected_assists, data['Assists_y'], test_size=0.2, random_state=42
    )

    model_goals = Ridge(alpha=2.0)
    model_goals.fit(X_train_goals, y_train_goals)
    y_pred_goals = model_goals.predict(X_test_goals)

    model_assists = Ridge(alpha=2.0)
    model_assists.fit(X_train_assists, y_train_assists)
    y_pred_assists = model_assists.predict(X_test_assists)


    mse_goals = mean_squared_error(y_test_goals, y_pred_goals)
    r2_goals = r2_score(y_test_goals, y_pred_goals)

    mse_assists = mean_squared_error(y_test_assists, y_pred_assists)
    r2_assists = r2_score(y_test_assists, y_pred_assists)


    print(f'Goals - MSE: {mse_goals}, R^2: {r2_goals}')
    print(f'Assists - MSE: {mse_assists}, R^2: {r2_assists}')

"""##Data Visualisation

###Scatter Plot of Actual v.s. Predicted Values
"""

results = pd.DataFrame({
    'Actual Goals': y_test['Goals_y'],
    'Predicted Goals': y_pred[:, 0],
    'Actual Assists': y_test['Assists_y'],
    'Predicted Assists': y_pred[:, 1]
})

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(data=results, x='Actual Goals', y='Predicted Goals', color='blue')
plt.plot([0, results['Actual Goals'].max()],
         [0, results['Actual Goals'].max()], 'r--')
plt.title('Actual vs Predicted Goals (Poisson Regression)')
plt.xlabel('Actual Goals')
plt.ylabel('Predicted Goals')
plt.axis('equal')
plt.xlim(0, results['Actual Goals'].max() + 1)
plt.ylim(0, results['Predicted Goals'].max() + 1)


plt.subplot(1, 2, 2)
sns.scatterplot(data=results, x='Actual Assists', y='Predicted Assists', color='green')
plt.plot([0, results['Actual Assists'].max()],
         [0, results['Actual Assists'].max()], 'r--')
plt.title('Actual vs Predicted Assists (Poisson Regression)')
plt.xlabel('Actual Assists')
plt.ylabel('Predicted Assists')
plt.axis('equal')
plt.xlim(0, results['Actual Assists'].max() + 1)
plt.ylim(0, results['Predicted Assists'].max() + 1)

plt.tight_layout()
plt.show()

"""###Residual Plots"""

results['Residual Goals'] = results['Actual Goals'] - results['Predicted Goals']
results['Residual Assists'] = results['Actual Assists'] - results['Predicted Assists']

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(data=results, x='Predicted Goals', y='Residual Goals', color='blue')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Goals')
plt.xlabel('Predicted Goals')
plt.ylabel('Residuals')
plt.xlim(0, results['Predicted Goals'].max() + 1)
plt.grid()

plt.subplot(1, 2, 2)
sns.scatterplot(data=results, x='Predicted Assists', y='Residual Assists', color='green')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Assists')
plt.xlabel('Predicted Assists')
plt.ylabel('Residuals')
plt.xlim(0, results['Predicted Assists'].max() + 1)
plt.grid()

plt.tight_layout()
plt.show()

"""###Histogram of Residuals"""

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.histplot(results['Residual Goals'], bins=30, kde=True, color='blue')
plt.title('Distribution of Residuals for Goals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid()

plt.subplot(1, 2, 2)
sns.histplot(results['Residual Assists'], bins=30, kde=True, color='green')
plt.title('Distribution of Residuals for Assists')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid()

plt.tight_layout()
plt.show()

"""###Regression Line Plots"""

plt.figure(figsize=(16, 32))

y_pred_df = pd.DataFrame(y_pred, columns=['Predicted Goals', 'Predicted Assists'])
results = pd.concat([preprocessed_data.reset_index(drop=True), y_pred_df.reset_index(drop=True)], axis=1)

selected_features = np.random.choice(preprocessed_data.columns[1:-2], size=4, replace=False)

for i, feature in enumerate(selected_features):
    plt.subplot(4, 2, i * 2 + 1)
    sns.regplot(data=results, x=feature, y='Goals_y', color='blue', label='Actual Goals', scatter_kws={'alpha':0.5})
    sns.regplot(data=results, x=feature, y='Predicted Goals', color='red', label='Predicted Goals', scatter_kws={'alpha':0.5})
    plt.title(f'Regression Line for Actual and Predicted Goals vs {feature}')
    plt.xlabel(feature)
    plt.ylabel('Goals')
    plt.legend()
    plt.grid()


    plt.subplot(4, 2, i * 2 + 2)
    sns.regplot(data=results, x=feature, y='Assists_y', color='green', label='Actual Assists', scatter_kws={'alpha':0.5})
    sns.regplot(data=results, x=feature, y='Predicted Assists', color='red', label='Predicted Assists', scatter_kws={'alpha':0.5})
    plt.title(f'Regression Line for Actual and Predicted Assists vs {feature}')
    plt.xlabel(feature)
    plt.ylabel('Assists')
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()

"""##Application"""

def predict_player_performance(player_name):
    matching_players = preprocessed_data[preprocessed_data['Player'].str.contains(player_name, case=False, na=False)]
    if matching_players.empty:
        return []

    performance_list = []

    for _, row in matching_players.iterrows():
        index_to_drop = preprocessed_data[(preprocessed_data['Player'] == row['Player'])].index
        X = preprocessed_data.drop(index=index_to_drop).iloc[:, 1:-2]
        y = preprocessed_data.drop(index=index_to_drop).iloc[:, -2:]
        model = MultiOutputRegressor(LinearRegression())
        model.fit(X, y)
        predicted = model.predict(matching_players.iloc[:, 1:-2])

        performance_list.append({
            'Player': row['Player'],
            'Actual Goals': row['Goals_y'],
            'Predicted Goals': predicted[0][0],
            'Actual Assists': row['Assists_y'],
            'Predicted Assists': predicted[0][1]
        })
    return performance_list

player_names = ["Bruno Fernandes"]
results = []
for name in player_names:
    performance = predict_player_performance(name)
    if not performance:
        results.append({'Player': name, 'Actual Goals': 'N/A', 'Predicted Goals': 'N/A',   'Actual Assists': 'N/A','Predicted Assists': 'N/A'})
    else:
        results.extend(performance)

performance_df = pd.DataFrame(results)
performance_df

"""#Classification

##Position Selection
"""

def filter_by_position(data, position_substring):
    if 'Pos' not in data.columns:
        return pd.Dataframe([])
    filtered_data = data[data['Pos'].str.contains(position_substring, case=False, na=False)]
    return filtered_data

data_pos = filter_by_position(data, 'FW')
data_pos

"""##Preprocessing"""

def preprocess_classification(data):
    data_without_player = data.drop(columns=['Player'], errors='ignore')
    numerical_data = data_without_player.select_dtypes(include=['number'])
    categorical_data_encoded = preprocess_categorical_data(data_without_player)
    categorical_data_encoded.reset_index(drop=True, inplace=True)
    numerical_data.reset_index(drop=True, inplace=True)
    combined_data = pd.concat([categorical_data_encoded, numerical_data], axis=1)
    data_imputed = impute_missing_values(combined_data)
    scaled_data = scale_numerical_data(data_imputed, "minmax")
    scaled_data.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    final_data = pd.concat([data['Player'], scaled_data.iloc[:, :-2]], axis=1)
    return final_data

preprocessed_data_classification = preprocess_classification(data_pos)
preprocessed_data_classification

"""##Classification Model

###K means
"""

kmeans_data = preprocessed_data_classification.iloc[:, 1:]
reference_players = []

feature_columns = kmeans_data.columns
for feature in feature_columns:
    low = kmeans_data[feature].quantile(0.1)
    lq = kmeans_data[feature].quantile(0.25)
    median = kmeans_data[feature].median()
    uq = kmeans_data[feature].quantile(0.75)
    top = kmeans_data[feature].quantile(0.9)

    reference_players.append([low, lq, median, uq, top])
reference_df = pd.DataFrame(reference_players, index=feature_columns)

kmeans = KMeans(n_clusters=5, init=reference_df.T.values, n_init=1, random_state=4211)
kmeans_data['Cluster'] = kmeans.fit_predict(kmeans_data)
cluster_means = kmeans_data.groupby('Cluster').mean()

from sklearn.metrics import silhouette_score
silhouette = silhouette_score(kmeans_data, kmeans_data['Cluster'])
print(f"Kmeans: {silhouette}")

performance_levels = ['Poor', 'Below Average', 'Average', 'Above Average', 'Elite']
cluster_ranking = cluster_means.mean(axis=1).sort_values().index
performance_mapping = {cluster: performance_levels[i] for i, cluster in enumerate(cluster_ranking)}
kmeans_data['Performance Level'] = kmeans_data['Cluster'].map(performance_mapping)

"""###Hierarchical Clustering"""

kmeans_data = preprocessed_data_classification.iloc[:, 1:]

hierarchical_model = AgglomerativeClustering(n_clusters=5)
kmeans_data['Cluster'] = hierarchical_model.fit_predict(kmeans_data)
hierarchical_means = kmeans_data.groupby('Cluster').mean()

silhouette = silhouette_score(kmeans_data, kmeans_data['Cluster'])
print(f"Hierarchical: {silhouette}")

performance_levels = ['Poor', 'Below Average', 'Average', 'Above Average', 'Elite']
cluster_ranking = cluster_means.mean(axis=1).sort_values().index
performance_mapping = {cluster: performance_levels[i] for i, cluster in enumerate(cluster_ranking)}
kmeans_data['Performance Level'] = kmeans_data['Cluster'].map(performance_mapping)

"""###Gaussian Mixture Model"""

kmeans_data = preprocessed_data_classification.iloc[:, 1:]

gmm_model = GaussianMixture(n_components=5)
kmeans_data['Cluster'] = gmm_model.fit_predict(kmeans_data)
gmm_means = kmeans_data.groupby('Cluster').mean()

silhouette_gmm = silhouette_score(kmeans_data, kmeans_data['Cluster'])
print(f"GMM: {silhouette_gmm}")

performance_levels = ['Poor', 'Below Average', 'Average', 'Above Average', 'Elite']
cluster_ranking = cluster_means.mean(axis=1).sort_values().index
performance_mapping = {cluster: performance_levels[i] for i, cluster in enumerate(cluster_ranking)}
kmeans_data['Performance Level'] = kmeans_data['Cluster'].map(performance_mapping)

"""##Data Visualisation

###Distribution of performances
"""

plt.figure(figsize=(10, 6))
sns.countplot(data=kmeans_data, x='Performance Level', palette='viridis')
plt.title('Distribution of Performance Levels')
plt.xlabel('Performance Level')
plt.ylabel('Number of Players')
plt.xticks(rotation=45)
plt.show()

"""###Features Plot"""

goals_assists = data[['Goals_x', 'Assists_x', 'MP']]

#print(goals_assists)
goals_assists['Goals_mp'] = goals_assists['Goals_x'] * goals_assists['MP']
goals_assists['Assists_mp'] = goals_assists['Assists_x'] * goals_assists['MP']

kmeans_data.reset_index(drop=True, inplace=True)
combined_data = pd.concat([kmeans_data, goals_assists[['Goals_mp', 'Assists_mp']].reset_index(drop=True)], axis=1)

plt.figure(figsize=(12, 8))
sns.scatterplot(data=combined_data, x='Goals_mp', y='Assists_mp', hue='Performance Level', s=100)
plt.xlim(0, combined_data['Goals_mp'].max() + 10)
plt.ylim(0, combined_data['Assists_mp'].max() + 10)
plt.title('Scatter Plot of Soccer Players: Goals vs. Assists (Scaled by MP)')
plt.xlabel('Goals (scaled by MP)')
plt.ylabel('Assists (scaled by MP)')
plt.legend(title='Performance Level')
plt.grid(True)

plt.show()

"""###PCA"""

pca = PCA(n_components=2)
pca_result = pca.fit_transform(kmeans_data.iloc[:, :-1])

pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = kmeans_data['Cluster']

plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100, alpha=0.7)
plt.title('Visualization using PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

"""###Silhouette Scores"""

#print(preprocessed_data_classification.columns)
kmeans_data = preprocessed_data_classification.iloc[:, 1:-2]

X_numeric = kmeans_data[['Squad', 'Comp']]

hierarchical_model = AgglomerativeClustering(n_clusters=5)
kmeans_data['Cluster'] = hierarchical_model.fit_predict(X_numeric)

silhouette_vals = silhouette_samples(X_numeric, kmeans_data['Cluster'])
kmeans_data['Silhouette Score'] = silhouette_vals


plt.figure(figsize=(10, 6))
sns.histplot(kmeans_data['Silhouette Score'], bins=30, kde=True, color='blue')
plt.title('Distribution of Silhouette Scores')
plt.xlabel('Silhouette Score')
plt.ylabel('Number of Players')
plt.axvline(silhouette_score(X_numeric, kmeans_data['Cluster']), color='red', linestyle='--', label='Average Silhouette Score')
plt.legend()
plt.grid(True)
plt.show()

"""##Application"""

def get_player_performance(player_name):

    matching_players = data[data['Player'].str.contains(player_name, case=False, na=False)]
    if matching_players.empty:
        return []

    performance_list = []

    for _, row in matching_players.iterrows():

        pos = data.loc[data['Player'] == row['Player'], 'Pos'].str[:2].values[0]
        data_pos = filter_by_position(data, pos)

        preprocessed_data_classification = preprocess_classification(data_pos)

        kmeans_data = preprocessed_data_classification.iloc[:, 1:]


        reference_players = []

        feature_columns = kmeans_data.columns

        for feature in feature_columns:
            low = kmeans_data[feature].quantile(0.1)
            lq = kmeans_data[feature].quantile(0.25)
            median = kmeans_data[feature].median()
            uq = kmeans_data[feature].quantile(0.75)
            top = kmeans_data[feature].quantile(0.9)

            reference_players.append([low, lq, median, uq, top])

        reference_df = pd.DataFrame(reference_players, index=feature_columns)

        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, init=reference_df.T.values, n_init=1, random_state=4211)
        kmeans_data['Cluster'] = kmeans.fit_predict(kmeans_data)
        cluster_means = kmeans_data.groupby('Cluster').mean()
        performance_levels = ['Poor', 'Below Average', 'Average', 'Above Average', 'Elite']

        cluster_ranking = cluster_means.mean(axis=1).sort_values().index
        performance_mapping = {cluster: performance_levels[i] for i, cluster in enumerate(cluster_ranking)}

        kmeans_data['Performance Level'] = kmeans_data['Cluster'].map(performance_mapping)

        performance_means = kmeans_data.groupby('Performance Level').mean()

        final_data = pd.concat([preprocessed_data_classification.iloc[:,0], kmeans_data], axis=1)

        performance_list.append({
            'Player': row['Player'],
            'Position': pos,
            'Performance Level': final_data.loc[final_data['Player'] == row['Player'], 'Performance Level'].values[0]
        })

    return performance_list


#player_names = ["kepa", "diger", "alonso", "jorginho", "Thiago Silva", "N'Golo", "Mateo Kova", "Pulisic", "Werner", "Ruben Loftus-Cheek", "Edouard Mendy","Mason Mount" , "Callum Hudson-Odoi", "Ben Chilwell", "Hakim Ziyech", "Reece James", "Azpilicueta"]
player_names = ["gallagher"]
results = []

for name in player_names:
    performance = get_player_performance(name)
    if not performance:
        results.append({'Player': name, 'Position': 'N/A', 'Performance Level': 'No performance data found'})
    else:
        results.extend(performance)

performance_df = pd.DataFrame(results)
performance_df

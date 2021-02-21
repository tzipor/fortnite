import ast
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

# ordered clusters to be used later
CLUSTER_COUNT = 22
REAL_NAMES = ['Salty Towers', 'Coral Castle', 'Catty Corner', 'Steamy Stacks', 'Misty Meadows', 'Middle Area', 'Pleasant Park', "Hunter's Haven", 'Hydro 16', 'Weeping Woods', 'West Coast', 'Frenzy Farm', 'Slurpy Swamp', 'Craggy Cliffs', 'Holly Hedges', 'Retail Row', 'Bottom Island', 'Dirty Docks', 'Stealthy Stronghold', 'Lazy Lake', 'Sweaty Sands', 'Colossal Coliseum']
z = []

# parse raw coordinates
with open('ChestCoordinates.txt', 'r') as txtfile:
    data = txtfile.read()
    data = data.replace('],[', ']x[')
    data = data.split('x')

    for thing in data:
        z.append(ast.literal_eval(thing))

# fix reversed coordinates
for thing in z:
    thing[0], thing[1] = thing[1], thing[0]

df = pd.DataFrame(z, columns=['x_coord', 'y_coord'])

# find chest clusters, name, and add to new column
cluster = AgglomerativeClustering(n_clusters=CLUSTER_COUNT, affinity='euclidean', linkage='ward')
labels = cluster.fit_predict(df)
named_labels = []
for coord in labels:
    named_labels.append(REAL_NAMES[coord])
df['location'] = pd.Series(named_labels, index = df.index)

sns.set()
fig, axes = plt.subplots(1, 2, figsize=(6,6))
img = plt.imread('Fortnite-Chapter-2-Season-5-Map.jpg')

# plot each chest on the game map
scatter = sns.scatterplot(ax=axes[0],
                          data = df,
                          x = 'x_coord',
                          y = 'y_coord',
                          edgecolor = 'none',
                          hue = 'location',
                          palette=sns.color_palette('hls', n_colors = CLUSTER_COUNT),
                          marker = 'd',
                          legend = False)
scatter.grid(False)
scatter.imshow(img, extent = [0,256,-256,0], zorder = 0)
scatter.set_ylabel('')
scatter.set_xlabel('')
scatter.set(xticklabels=[])
scatter.set(yticklabels=[])

# show sorted counts of chests at each location
bar = sns.countplot(ax=axes[1],
                    data = df,
                    y = df['location'],
                    palette = sns.color_palette('hls', n_colors = CLUSTER_COUNT),
                    edgecolor = 'none',
                    order = df['location'].value_counts().index,
                    hue = df['location'],
                    dodge = False)
bar.grid(False)
bar.set(xlabel='', ylabel='')
bar.set(xticklabels=[])
bar.set(yticklabels=[])
for p in bar.patches:
    bar.annotate(p.get_width(), xy=(p.get_width()-p.get_width() + 0.05, p.get_y()+p.get_height()/2 + 0.05),xytext=(5, 0), fontsize=18, textcoords='offset points', ha="left", va="center", color = 'white')

plt.subplots_adjust(wspace = 0)
fig.tight_layout()
fig.suptitle('Chest Locations in Fortnite', size = 24)
plt.subplots_adjust(top=0.85)
plt.show()
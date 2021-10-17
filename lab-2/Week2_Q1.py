import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% read dataset
df = pd.read_csv("pokemon0820.csv")
print(df)


# %% define function for sorting the pokemon
def sort(against: str) -> pd.DataFrame:
    """
    Sort the pokemon by resistivity to the given category of pokemon
    For the pokemon with top 6 resistivity, their corresponding 'point' column
    is increased by 1

    :arg against The category to which the pokemon is against which is being
    sorted
    :return The top 6 pokemon
    """

    label = 'against_' + against
    dff = df.sort_values(label, ascending=False)
    index = dff.index
    df.loc[index[:6], 'point'] += 1
    return dff.head(6)[[label, 'japanese_name', 'name']]


# %% sort by resistivity
# obtains the name of categories
labels = [label[8:] for label in df.columns[1:19]]
df['point'] = 0  # initialize the point column
for label in labels:
    print('Selecting the 6 pokemon with top resistivity against', label)
    print(sort(label))
    print()

point_max = df['point'].max()
print('The maximum point is ', point_max)
point_top = df[df['point'] == point_max]

# %% sort by hp
choice = point_top.sort_values('hp', ascending=False).head(6)
print(choice)
choice.describe()

# %% plot
fig = plt.figure(figsize=(9, 6))
# plt.title('My Pokemon Choice')
choice_reindex = choice.reset_index(drop=True)
theta = [2 * np.pi / 5 * i for i in range(6)] + [0]
labels = ['sp_defense', 'sp_attack', 'defense', 'attack', 'hp', 'speed']

for i in range(6):
    ax = plt.subplot(2, 3, i + 1, polar=True)
    pokemon = choice_reindex.loc[i]
    name = pokemon['name']
    r = pokemon[labels + [labels[0]]].tolist()
    ax.plot(theta, r)
    ax.set_thetagrids(np.array(theta) * 180 / np.pi, labels + [''])
    ax.set_rlim(0, 250)
    plt.title(name)
fig.tight_layout()
# plt.savefig('pokemon.png')
plt.show()

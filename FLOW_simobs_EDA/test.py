import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a sample dataset
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

# Create the plot
ax = sns.lineplot(data=df, x='x', y='y')

# Change tick labels to include superscript text
new_labels = [r'5x$10^{'+str(x)+r'}$' for x in [-1,-2,-3]]
ax.set_xticklabels(new_labels)

plt.show()
exit(0)
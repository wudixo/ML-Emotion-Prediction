import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("emotional_impact_dataset.csv")

# Define stable and unstable groups
df['emotional_state'] = df['stable_emotional_state'].map({1: 'Stable', 0: 'Unstable'})

# Calculate mean values for each group
group_means = df.groupby('emotional_state')[['work_productivity', 'mental_health', 'social_interaction', 'physical_health']].mean()

# Plot the comparisons
group_means.plot(kind='bar', figsize=(10,6))
plt.title('Impact of Emotional Stability on Life Domains')
plt.ylabel('Average Score')
plt.xticks(rotation=0)
plt.legend(title='Life Domain')
plt.tight_layout()
plt.show()

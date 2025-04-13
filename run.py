import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed and dataset size
np.random.seed(42)
n = 1000

# Simulate base data
relationship_conflict = np.random.randint(0, 11, size=n)
support_from_partner = np.random.randint(0, 11, size=n)
emotional_stability = 10 - (relationship_conflict * 0.5) + (support_from_partner * 0.3)
emotional_stability = np.clip(emotional_stability + np.random.normal(0, 1, size=n), 1, 10)

# Derived metrics
work_productivity = np.clip(emotional_stability + np.random.normal(0, 1.5, size=n), 1, 10)
mental_health = np.clip(emotional_stability + np.random.normal(0, 1.0, size=n), 1, 10)
social_interaction = np.clip(emotional_stability + np.random.normal(0, 1.2, size=n), 1, 10)
physical_health = np.clip(emotional_stability + np.random.normal(0, 1.3, size=n), 1, 10)
relationship_with_children = np.clip(emotional_stability + np.random.normal(0, 1.4, size=n), 1, 10)
relationship_with_friends = np.clip(emotional_stability + np.random.normal(0, 1.2, size=n), 1, 10)
relationship_with_family = np.clip(emotional_stability + np.random.normal(0, 1.1, size=n), 1, 10)
study_performance = np.clip(emotional_stability + np.random.normal(0, 1.6, size=n), 1, 10)
risk_of_job_loss = np.clip(10 - work_productivity + np.random.normal(0, 1), 0, 10)

# Emotional state label
stable_emotional_state = (emotional_stability >= 6).astype(int)

# Combine into DataFrame
df = pd.DataFrame({
    'emotional_stability': emotional_stability,
    'relationship_conflict': relationship_conflict,
    'support_from_partner': support_from_partner,
    'work_productivity': work_productivity,
    'mental_health': mental_health,
    'social_interaction': social_interaction,
    'physical_health': physical_health,
    'relationship_with_children': relationship_with_children,
    'relationship_with_friends': relationship_with_friends,
    'relationship_with_family': relationship_with_family,
    'study_performance': study_performance,
    'risk_of_job_loss': risk_of_job_loss,
    'stable_emotional_state': stable_emotional_state
})

# Convert 0/1 to labels
df['emotional_state'] = df['stable_emotional_state'].map({1: 'Stable', 0: 'Unstable'})

# Group by emotional state
grouped = df.groupby('emotional_state')[
    ['relationship_with_children', 'relationship_with_friends', 'relationship_with_family',
     'work_productivity', 'risk_of_job_loss', 'physical_health', 'mental_health', 'study_performance']
].mean()

# Plot 1: Positive Impacts
positive_metrics = ['relationship_with_children', 'relationship_with_friends', 'relationship_with_family',
                    'work_productivity', 'physical_health', 'mental_health', 'study_performance']

plt.figure(figsize=(12, 6))
grouped.loc[:, positive_metrics].T.plot(kind='bar')
plt.title('Positive Life Impacts by Emotional Stability')
plt.ylabel('Average Score (Higher is Better)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Negative Impact (Job Loss Risk)
plt.figure(figsize=(6, 4))
grouped['risk_of_job_loss'].plot(kind='bar', color=['red', 'green'])
plt.title("Risk of Job Loss by Emotional Stability")
plt.ylabel("Risk Score (Higher = Worse)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
